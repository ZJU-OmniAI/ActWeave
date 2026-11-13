import { Kinematics,type Side } from './kinematics.ts';
import { orientation,position,quatPose,norm,sub,mul,axisRotation,rotationError,angles,type V,type M } from './math.ts';
export class Planner {
 constructor(public kin:Kinematics,public initial:Record<Side,V>) {}
 plan(side:Side,target:M,start:V,pathMode:'auto'|'line'|'joint') {
  const goal=this.kin.solve(side,target,start,this.initial[side]);
  if(!goal.reached)return {ik:goal,path:pathMode,reason:'ik_unreachable',message:`No accepted endpoint IK solution. Best solver residual: ${goal.position_error_m.toFixed(6)} m and ${(goal.rotation_error_rad*180/Math.PI).toFixed(4)} degrees.`,waypoints:[]};
  if(pathMode!=='joint') {
   const begin=this.kin.tcp(side,start),turn=rotationError(begin,target);
   const count=Math.max(1,Math.ceil(norm(sub(position(target),position(begin)))/0.02),Math.ceil(norm(turn)/(8*Math.PI/180)));
   const waypoints:V[]=[];let q=start;
   for(let i=1;i<=count;i++) {
    const fraction=i/count;
    const pose=norm(turn)>1e-8?mul(axisRotation(turn,norm(turn)*fraction),begin):begin.map(row=>[...row]);
    for(let k=0;k<3;k++)pose[k][3]=begin[k][3]+fraction*(target[k][3]-begin[k][3]);
    const solution=this.kin.solveFrom(side,pose,q);
    if(!solution.reached){
     if(pathMode==='line')return {ik:goal,path:'line',reason:'line_unreachable',message:`No accepted IK solution for line waypoint ${i} of ${count}; no part of the route was executed.`,waypoint_ik:solution,failed_waypoint:i-1,waypoints:[]};
     return {ik:goal,path:'joint',reason:null,message:null,line_failure:{waypoint:i-1,ik:solution},waypoints:[goal.joints]};
    }
    q=solution.joints;waypoints.push(q);
   }
   return {ik:goal,path:'line',reason:null,message:null,waypoints};
  }
  return {ik:goal,path:'joint',reason:null,message:null,waypoints:[goal.joints]};
 }
 trajectory(side:Side,targets:M[],start:V) {
  const waypoints:V[]=[],segments=[];let current=start;
  for(let index=0;index<targets.length;index++) {
   const plan=this.plan(side,targets[index],current,'line');
   if(plan.reason)return {reason:plan.reason,message:plan.message,failed_segment:index,failed_plan:plan,waypoints:[],segments};
   waypoints.push(...plan.waypoints);segments.push({index,waypoint_end:waypoints.length,target:targets[index],ik:plan.ik});
   if(plan.waypoints.length)current=plan.waypoints[plan.waypoints.length-1];
  }
  return {reason:null,message:null,waypoints,segments};
 }
 prepareMove(side:Side,xyz:V,pitch:number|undefined,yaw:number|undefined,roll:number|undefined,start:any,pathMode:'auto'|'line'|'joint') {
  const currentPose=quatPose(start[side].link_pose),currentAngles=angles(currentPose);
  const explicitOrientation=[pitch,yaw,roll].some(v=>v!==undefined);
  const selectedPitch=pitch??currentAngles[0],selectedYaw=yaw??currentAngles[1];
  const selectedRoll=roll??(Math.abs(Math.cos(selectedPitch*Math.PI/180))<Math.sin(Math.PI/180)?0:currentAngles[2]);
  const target=explicitOrientation?orientation(selectedPitch,selectedYaw,selectedRoll):currentPose;
  xyz.forEach((x,i)=>target[i][3]=x);
  return {plan:this.plan(side,target,start[side].joints,pathMode),target};
 }
}
