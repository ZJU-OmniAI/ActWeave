import { Worker } from 'node:worker_threads';
import { Episode } from '../bridge/episode.ts';
import { Kinematics,type Side } from './kinematics.ts';
import { position,quatPose,apply,norm,sub,rotationError,angles,type V } from './math.ts';
export class Robot {
 openings:Record<Side,number>;
 initial:Record<Side,V>;
 constructor(public episode:Episode,public kin:Kinematics) {
  const state=episode.read().state;
  this.initial={left:[...state.left_arm_joint_state],right:[...state.right_arm_joint_state]};
  this.openings={left:state.left_ee_joint_state[0],right:state.right_ee_joint_state[0]};
 }
 state(step=-1) {
  const obs=this.episode.read(step);const arms:any={};
  for(const side of ['left','right'] as Side[]) {
   const pose=quatPose(obs.state[side+'_ee_pose']);
   arms[side]={joints:obs.state[side+'_arm_joint_state'],link_pose:obs.state[side+'_ee_pose'],tcp:apply(pose,[this.kin.config.robot.gripper_bias,0,0]),gripper_opening:obs.state[side+'_ee_joint_state'][0]};
  }
  return {step:obs.step,step_limit:obs.step_limit,ended:this.episode.ended,planning_joint_ranges_rad:this.kin.chain.map(j=>j.limits),...arms};
 }
 publicState(step=-1) {
  const state=this.state(step),arms:any={};
  for(const side of ['left','right'] as Side[]) {
   const arm=state[side],[pitch,yaw,roll]=angles(quatPose(arm.link_pose));
   arms[side]={joints:arm.joints,tcp_pos:arm.tcp,eef_pos:arm.link_pose.slice(0,3),eef_quat:arm.link_pose.slice(3),pitch,yaw,roll,gripper:arm.gripper_opening,requested_opening:step<0||step===this.episode.read().step?this.openings[side]:null};
  }
  return {step:state.step,step_limit:state.step_limit,task:this.episode.read(step).instruction,truncated:state.step>=state.step_limit,ended:state.ended,...arms};
 }
 command():Record<string,number[]> { const last=this.episode.lastAction;return {left_arm_joint_state:[...(last?.left_arm_joint_state??this.initial.left)],right_arm_joint_state:[...(last?.right_arm_joint_state??this.initial.right)],left_ee_joint_state:[this.openings.left],right_ee_joint_state:[this.openings.right]}; }
 motionFeedback(before:ReturnType<Robot['publicState']>,callerMaxSteps:number|undefined,reason:string|null) {
  const state=this.publicState(),command=this.command();
  const arms=Object.fromEntries((['left','right'] as Side[]).map(side=>{
   const measured=state[side],target=command[side+'_arm_joint_state'],error=sub(target,measured.joints);
   return [side,{tcp_start_m:before[side].tcp_pos,tcp_end_m:measured.tcp_pos,tcp_delta_m:sub(measured.tcp_pos,before[side].tcp_pos),
    measured_joints_rad:measured.joints,commanded_joints_rad:target,joint_error_rad:error,max_joint_error_rad:Math.max(...error.map(Math.abs))}];
  }));
  return {start_step:before.step,end_step:state.step,executed_steps:state.step-before.step,caller_max_steps:callerMaxSteps??null,
   native_steps_remaining:Math.max(0,state.step_limit-state.step),limit_kind:reason==='step_budget'?(state.truncated?'native_action_limit':'caller_max_steps'):null,arms};
 }
 planning(method:'plan'|'prepareMove'|'trajectory',args:any[],signal?:AbortSignal):Promise<any> {
  const cancellation=signal?AbortSignal.any([signal,this.episode.stopped.signal]):this.episode.stopped.signal;
  cancellation.throwIfAborted();
  return new Promise((resolve,reject)=>{
   const worker=new Worker(new URL('./worker-entry.mjs',import.meta.url),{workerData:{method,args,initial:this.initial,config:this.kin.config,chain:this.kin.chain}});
   const cleanup=()=>{cancellation.removeEventListener('abort',abort);worker.removeAllListeners('exit');void worker.terminate();};
   const abort=()=>{cleanup();reject(new Error('Motion planning cancelled'));};
   cancellation.addEventListener('abort',abort,{once:true});
   worker.once('message',result=>{cleanup();resolve(result);});
   worker.once('error',error=>{cleanup();reject(error);});
   worker.once('exit',code=>{cleanup();reject(new Error(`Motion planner exited before returning a result (${code})`));});
  });
 }
 async follow(side:Side,waypoints:V[],maxSteps:number,signal?:AbortSignal,maxJointStep=0.15) {
  let steps=0,completed=0,waypointSteps=0,tolerance:number|null=null;
  const feedback=(reason:string|null,pathComplete:boolean)=>({steps,waypoints_completed:completed,path_complete:pathComplete,reason,
   tracking:{waypoints_total:waypoints.length,active_waypoint:pathComplete?null:completed,active_waypoint_steps:pathComplete?null:waypointSteps,
    active_joint_error_rad:pathComplete?null:Math.max(...sub(waypoints[completed],this.state()[side].joints).map(Math.abs)),active_tolerance_rad:pathComplete?null:tolerance}});
  for(const goal of waypoints) {
   const initialError=Math.max(...sub(goal,this.state()[side].joints).map(Math.abs));
   tolerance=Math.max(1e-8,Math.min(0.001,initialError*0.1));waypointSteps=0;
   while(true) {
    if(steps>=maxSteps||this.episode.read().step>=this.episode.read().step_limit)return feedback('step_budget',false);
    const current=this.state()[side].joints as V,delta=sub(goal,current),error=Math.max(...delta.map(Math.abs));
    const action=this.command(),fraction=error===0?1:Math.min(1,maxJointStep/error);
    action[side+'_arm_joint_state']=current.map((q,i)=>q+delta[i]*fraction);
    await this.episode.action(action,signal);steps++;waypointSteps++;
    const remaining=Math.max(...sub(goal,this.state()[side].joints).map(Math.abs));
    if(remaining<=tolerance)break;
   }
   completed++;
  }
  return feedback(null,true);
 }
 async move(side:Side,xyz:V,pitch:number|undefined,yaw:number|undefined,roll:number|undefined,maxSteps:number,signal?:AbortSignal,pathMode:'auto'|'line'|'joint'='auto',gripper=0,maxJointStep=0.15) {
  const {plan,target}=await this.planning('prepareMove',[side,xyz,pitch,yaw,roll,this.state(),pathMode],signal);
  signal?.throwIfAborted();
  this.episode.stopped.signal.throwIfAborted();
  if(!plan.reason&&gripper!==0)this.openings[side]=gripper===1?0:1;
  const execution=plan.reason?{steps:0,waypoints_completed:0,path_complete:false,reason:null,tracking:null}:await this.follow(side,plan.waypoints,maxSteps,signal,maxJointStep);
  const actual=this.state(),positionError=norm(sub(actual[side].tcp,xyz)),angle=norm(rotationError(quatPose(actual[side].link_pose),target));
  const finalGoal=plan.waypoints.at(-1),jointError=finalGoal?Math.max(...sub(finalGoal,actual[side].joints).map(Math.abs)):null;
  const reason=plan.reason??execution.reason??(positionError>0.002||angle>Math.PI/180?'tracking_residual':null);
  return {reached:reason===null&&execution.path_complete,reason,position_error_m:positionError,rotation_error_rad:angle,max_joint_error_rad:jointError,
   steps:execution.steps,waypoints_completed:execution.waypoints_completed,path_complete:execution.path_complete,tracking:execution.tracking,path:plan.reason?null:plan.path,plan,target_pose:target,state:this.publicState()};
 }
 async wait(steps:number,captureAt:number[],signal?:AbortSignal) {
  const start=this.episode.read().step,command=structuredClone(this.episode.lastAction??this.command()),frames:number[]=[];
  if(captureAt.includes(0))frames.push(start);
  let reason='requested_steps';
  try {
   for(let used=0;used<steps;used++){
    signal?.throwIfAborted();
    if(this.episode.ended){reason=this.episode.reason;break;}
    if(this.episode.read().step>=this.episode.read().step_limit){reason='step_budget';break;}
    await this.episode.action(command,signal);
    if(captureAt.includes(this.episode.read().step-start))frames.push(this.episode.read().step);
   }
  } catch(error) {
   reason=signal?.aborted?'cancelled':this.episode.ended?this.episode.reason:'';
   if(!reason)throw error;
  }
  const state=this.publicState();
  if(captureAt.includes(state.step-start)&&!frames.includes(state.step))frames.push(state.step);
  if(this.episode.ended)reason=this.episode.reason;
  else if(state.truncated)reason='step_budget';
  return {requested_steps:steps,steps_used:state.step-start,start_step:start,step:state.step,reason,frames,command,state};
 }
 async grip(side:Side,opening:number,steps:number,signal?:AbortSignal) {
  this.openings[side]=opening;
  let used=0,last=this.state()[side].gripper_opening,stable=false;
  for(let i=0;i<steps;i++){
   const action=this.command();action[side+'_ee_joint_state']=[opening];await this.episode.action(action,signal);used++;
   const now=this.state()[side].gripper_opening;stable=Math.abs(now-last)<0.004;last=now;
  }
  const state=this.state(),measured=state[side].gripper_opening,holding=opening===0&&measured>0.01;
  return {arm:side,commanded:opening===0?'closed':'open',opening:measured,holding,width_m:0.0023+0.1091*measured,stable,steps_used:used,step:state.step,step_limit:state.step_limit,truncated:state.step>=state.step_limit};
 }
 async planPose(side:Side,target:number[][],pathMode:'auto'|'line'|'joint',signal?:AbortSignal) {
  return this.planning('plan',[side,target,this.state()[side].joints,pathMode],signal);
 }
 async movePose(side:Side,target:number[][],maxSteps:number,pathMode:'auto'|'line'|'joint',gripper:number,signal?:AbortSignal,maxJointStep=0.15) {
  const before=this.publicState(),plan=await this.planPose(side,target,pathMode,signal);
  if(!plan.reason&&gripper!==0)this.openings[side]=gripper===1?0:1;
  const execution=plan.reason?{steps:0,waypoints_completed:0,path_complete:false,reason:null,tracking:null}:await this.follow(side,plan.waypoints,maxSteps,signal,maxJointStep);
  const state=this.publicState(),actual=this.state()[side];
  const positionError=norm(sub(actual.tcp,position(target))),orientationError=norm(rotationError(quatPose(actual.link_pose),target))*180/Math.PI;
  return {arm:side,target_pose:target,reached:!plan.reason&&execution.path_complete&&positionError<=0.002&&orientationError<=1,
   reason:plan.reason??execution.reason??(positionError>0.002||orientationError>1?'tracking_residual':null),position_error_m:positionError,orientation_error_deg:orientationError,
   plan,steps_used:execution.steps,waypoints_completed:execution.waypoints_completed,path_complete:execution.path_complete,tracking:execution.tracking,
   max_joint_error_rad:plan.waypoints.length?Math.max(...sub(plan.waypoints.at(-1)!,actual.joints).map(Math.abs)):null,before,state,state_step:before.step,
   physical_contact:null,retention:null};
 }
 async moveTrajectory(side:Side,targets:number[][][],maxSteps:number,signal?:AbortSignal,maxJointStep=0.15) {
  const before=this.publicState(),plan=await this.planning('trajectory',[side,targets,this.state()[side].joints],signal);
  const execution=plan.reason?{steps:0,waypoints_completed:0,path_complete:false,reason:null,tracking:null}:await this.follow(side,plan.waypoints,maxSteps,signal,maxJointStep);
  const target=targets[targets.length-1],actual=this.state()[side],positionError=norm(sub(actual.tcp,position(target))),orientationError=norm(rotationError(quatPose(actual.link_pose),target))*180/Math.PI;
  return {arm:side,reached:!plan.reason&&execution.path_complete&&positionError<=0.002&&orientationError<=1,reason:plan.reason??execution.reason??(positionError>0.002||orientationError>1?'tracking_residual':null),
   position_error_m:positionError,orientation_error_deg:orientationError,steps_used:execution.steps,waypoints_completed:execution.waypoints_completed,path_complete:execution.path_complete,tracking:execution.tracking,
   max_joint_error_rad:plan.waypoints.length?Math.max(...sub(plan.waypoints.at(-1)!,actual.joints).map(Math.abs)):null,plan,before,state:this.publicState(),
   state_step:before.step,physical_contact:null,retention:null};
 }
 async home(side:Side|'both',steps:number,signal?:AbortSignal,lift=false,maxJointStep?:number) {
  const sides:Side[]=side==='both'?['left','right']:[side],lifted:Record<string,number>={},liftPlans:any={};
  let used=0;
  const jointError=()=>Math.max(...sides.flatMap(name=>sub(this.state()[name].joints,this.initial[name]).map(Math.abs)));
  const feedback=(reason:string|null)=>({arm:side,reached:reason===null,max_joint_error_rad:jointError(),steps:used,lifted_m:lifted,lift_plans:liftPlans,reason,state:this.publicState(),truncated:this.state().step>=this.state().step_limit});
  for(const name of sides) {
   const start=this.state();lifted[name]=0;
   if(!lift)continue;
   const target=quatPose(start[name].link_pose);
   start[name].tcp.forEach((v:number,i:number)=>target[i][3]=v);target[2][3]+=0.06;
   const plan=await this.planning('plan',[name,target,start[name].joints,'line'],signal);liftPlans[name]=plan;
   if(plan.reason)return feedback(plan.reason);
   const result=await this.follow(name,plan.waypoints,steps-used,signal,maxJointStep??0.15);used+=result.steps;
   lifted[name]=this.state()[name].tcp[2]-start[name].tcp[2];
   if(result.reason)return feedback(result.reason);
  }
  let commanded=false;
  while(!commanded||jointError()>0.001) {
   if(used>=steps||this.episode.read().step>=this.episode.read().step_limit)return feedback('step_budget');
   const action=this.command();
   for(const name of sides) {
    const current=this.state()[name].joints,delta=sub(this.initial[name],current),error=Math.max(...delta.map(Math.abs));
    const fraction=maxJointStep===undefined||error===0?1:Math.min(1,maxJointStep/error);
    action[name+'_arm_joint_state']=maxJointStep===undefined?[...this.initial[name]]:current.map((q:number,i:number)=>q+delta[i]*fraction);
   }
   await this.episode.action(action,signal);used++;commanded=true;
  }
  return feedback(null);
 }
}
