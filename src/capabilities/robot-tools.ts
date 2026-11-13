import {Worker} from 'node:worker_threads';
import type {Register} from '../episode/execution.ts';
import type {EpisodeEvidence} from '../episode/evidence.ts';
import type {AstraPresentation} from '../agent/presentation.ts';

export function registerRobotCapabilities(tool:Register,evidence:EpisodeEvidence,presentation:AstraPresentation,budgetSnapshot:()=>Promise<any>) {
 const robot=evidence.robot,vision=evidence.vision,episode=evidence.episode;
 const motion:Register=(name,description,parameters,execute,render)=>tool(name,description,parameters,execute,render,{kind:'motion'});
 const joint:Register=(name,description,parameters,execute,render)=>tool(name,description,parameters,execute,render,{kind:'motion',rawMotion:true});
 const measure:Register=(name,description,parameters,execute,render)=>tool(name,description,parameters,execute,render,{kind:'measurement',sourceSteps:(args,start)=>[args.masks?.[0]?.step??start]});
 const side={type:'string',enum:['left','right'],required:true};
 const camera={type:'string',enum:['cam_head','cam_left_wrist','cam_right_wrist'],required:true};
 const xyz={type:'array',items:{type:'number'},required:true};
 tool('budget','Read a sampled snapshot of actual remaining wall/cell time and existing request, upstream attempt, cell and native action allowances. This does not step physics or replenish any budget. Missing router readings are explicit.',{},budgetSnapshot);
 tool('observation','Read public state and native instruction at a recorded native step, default current, without displaying images.',{step:{type:'integer'}},(a:any)=>evidence.state(a.step));
 tool('image','Read full PNG image data for Python calculations; no image is displayed to the model.',{camera,step:{type:'integer'}},(a:any)=>evidence.image(a.camera,a.step));
 tool('show','Display the selected public camera to the model.',{camera,step:{type:'integer'}},(a:any)=>presentation.show(a.camera,a.step),
  (v:any)=>[{type:'text',text:`${v.camera} at step ${v.step}`},{type:'image',attachment:v.attachment}]);
 tool('camera','Read public camera calibration and pose. transform maps camera coordinates to world; inverse maps world to camera. Camera axes: +x right, +y up, forward -z. Pixel row increases down, column right; intrinsics are in pixels. Optional step reads a recorded public pose; default is current.',{camera,step:{type:'integer'}},(a:any)=>evidence.calibration(a.camera,a.step));
 measure('segment','Segment an object by text, pixel point [row,col], multiple points with 0/1 labels, or bbox. Returns candidate masks and scores; neither establishes identity or a grasp pose.',{camera,text:{type:'string'},point:{type:'array',items:{type:'integer'}},points:{type:'array',items:{type:'array',items:{type:'integer'}}},labels:{type:'array',items:{type:'integer'}},box:{type:'array',items:{type:'number'}},min_score:{type:'number'}},(a:any,e:any)=>vision.segment(a.camera,a,e.signal,evidence.raw()));
 tool('display_image','Display a PNG computed by your Python program, such as a crop or a contact sheet.',{png_base64:{type:'string',required:true}},(a:any)=>presentation.display(a.png_base64),
  (v:any)=>[{type:'image',attachment:v.attachment}]);
 const segmentPrompt={oneOf:[
  {type:'object',additionalProperties:false,properties:{text:{type:'string',required:true},min_score:{type:'number'}}},
  {type:'object',additionalProperties:false,properties:{point:{type:'array',items:{type:'integer'},required:true},min_score:{type:'number'}}},
  {type:'object',additionalProperties:false,properties:{points:{type:'array',items:{type:'array',items:{type:'integer'}},required:true},labels:{type:'array',items:{type:'integer'}},min_score:{type:'number'}}},
  {type:'object',additionalProperties:false,properties:{box:{type:'array',items:{type:'number'},required:true},min_score:{type:'number'}}}
 ]};
 const viewPrompts={type:'object',additionalProperties:false,properties:{cam_head:segmentPrompt,cam_left_wrist:segmentPrompt,cam_right_wrist:segmentPrompt}};
 const capGeometry=(method:'locate'|'back_project',query:any,signal?:AbortSignal)=>{
  const observation=evidence.raw(query.mask_step),view=vision.camera(query.camera,observation.step);
  if(!Number.isFinite(query.support_z)||query.region_index!==undefined&&query.region_index<0)throw new Error('Finite support_z and nonnegative region_index are required');
  const pixel=method==='back_project'?[query.row,query.col]:query.prompt.point;
  if(pixel&&(pixel.length!==2||!pixel.every(Number.isInteger)||pixel[0]<0||pixel[0]>=view.height||pixel[1]<0||pixel[1]>=view.width))throw new Error('point requires an original pixel within the selected camera');
  const cancellation=AbortSignal.any([episode.stopped.signal,...(signal?[signal]:[])]);
  cancellation.throwIfAborted();
  return new Promise((resolve,reject)=>{
   const worker=new Worker(new URL('../vision/worker-entry.mjs',import.meta.url),{workerData:{observation,config:robot.kin.config,chain:robot.kin.chain,geometry:robot.kin.geometry,url:vision.url,method,query,heights:Array.from(vision.heights),surfaces:vision.surfaces}});
   const cleanup=()=>{clearTimeout(timer);cancellation.removeEventListener('abort',abort);worker.removeAllListeners('exit');void worker.terminate();};
   const abort=()=>{cleanup();reject(new Error('CAP geometry cancelled'));};
   const timer=setTimeout(()=>{cleanup();reject(new Error('CAP geometry deadline'));},120000);
   cancellation.addEventListener('abort',abort,{once:true});
   worker.once('message',value=>{cleanup();if(observation.step===episode.read().step){vision.heights=new Map(value.heights);vision.surfaces=value.surfaces;}resolve(value.result);});
   worker.once('error',error=>{cleanup();reject(error);});
   worker.once('exit',code=>{cleanup();reject(new Error(`CAP geometry exited before returning a result (${code})`));});
  });
 };
 measure('locate','Fit CAP geometry from segmentation prompts or explicit binary PNG masks at one recorded step. Mask mode uses only supplied cameras and pixels without SAM3 or identity matching. Returns fitted geometry and evidence, not object truth.',{camera,prompt:segmentPrompt,support_z:{type:'number'},view_prompts:viewPrompts,region_index:{type:'integer'},masks:{type:'array',items:{type:'object',additionalProperties:false,properties:{camera,step:{type:'integer',required:true},png_base64:{type:'string',required:true}}}}},(a:any,e:any)=>{
  if(a.masks!==undefined) {
   if(a.prompt!==undefined||a.view_prompts!==undefined||a.region_index!==undefined||!a.masks.length)throw new Error('Supply masks or segmentation prompts, not both');
   const steps=new Set(a.masks.map((mask:any)=>mask.step)),names=new Set(a.masks.map((mask:any)=>mask.camera));
   const mask_step=a.masks[0].step;
   if(steps.size!==1||mask_step<0||mask_step>episode.read().step||names.size!==a.masks.length||!names.has(a.camera))throw new Error('Masks require distinct cameras including the anchor and one existing observation step');
   return capGeometry('locate',{camera:a.camera,prompt:{},support_z:a.support_z??0.765,region_index:0,mask_step,masks:Object.fromEntries(a.masks.map((mask:any)=>[mask.camera,mask.png_base64]))},e.signal);
  }
  if(!a.prompt)throw new Error('A segmentation prompt or selected masks is required');
  return capGeometry('locate',{...a,support_z:a.support_z??0.765,view_prompts:a.view_prompts??{},region_index:a.region_index??0},e.signal);
 });
 measure('back_project','Intersect a current pixel ray with CAP-fitted geometry or a labelled support-plane estimate. surface/note identify the estimate; xyz can be unresolved.',{camera,row:{type:'integer',required:true},col:{type:'integer',required:true},support_z:{type:'number'}},(a:any,e:any)=>capGeometry('back_project',{...a,support_z:a.support_z??0.765},e.signal));
 motion('wait','Repeat the current dual-arm and gripper control command for the requested number of native actions. capture_at lists relative action offsets whose original public frames remain available. No time scaling or gripper change.',{steps:{type:'integer',required:true},capture_at:{type:'array',items:{type:'integer'},required:true}},(a:any,e:any)=>{
  if(!Number.isInteger(a.steps)||a.steps<0||!a.capture_at.every((n:number)=>Number.isInteger(n)&&n>=0&&n<=a.steps))throw new Error('steps and capture_at must be nonnegative native action counts within this call');
  return robot.wait(a.steps,a.capture_at,e.signal);
 });
 joint('joint_action','Send one complete dual-arm joint action and return the next public state. Use for custom trajectories; one native step.',{left:{type:'array',items:{type:'number'},required:true},right:{type:'array',items:{type:'number'},required:true},left_opening:{type:'number',required:true},right_opening:{type:'number',required:true}},async(a:any,e:any)=>{if(a.left.length!==6||a.right.length!==6||a.left_opening<0||a.left_opening>1||a.right_opening<0||a.right_opening>1)throw new Error('Each arm requires six joint angles');robot.openings.left=a.left_opening;robot.openings.right=a.right_opening;await episode.action({left_arm_joint_state:a.left,right_arm_joint_state:a.right,left_ee_joint_state:[a.left_opening],right_ee_joint_state:[a.right_opening]},e.signal);return robot.publicState();});
 measure('ray_plane','Intersect a pixel ray with an explicitly supplied world z plane. The result is a surface point, not a TCP.',{camera,row:{type:'number',required:true},col:{type:'number',required:true},z:{type:'number',required:true}},(a:any)=>vision.rayPlane(a.camera,a.row,a.col,a.z,evidence.step));
 const jointStep={type:'number',description:'Optional maximum commanded joint increment per native action in radians, >0 and <=0.15. Not Cartesian speed or force control.'};
 const jointIncrement=(a:any)=>{if(a.max_joint_step_rad!==undefined&&(!Number.isFinite(a.max_joint_step_rad)||a.max_joint_step_rad<=0||a.max_joint_step_rad>0.15))throw new Error('max_joint_step_rad must be in (0,0.15] radians');return a.max_joint_step_rad;};
 const pose={type:'array',items:{type:'array',items:{type:'number'}},required:true};
 const validPose=(matrix:any)=>{
  if(matrix.length!==4||matrix.some((row:any)=>row.length!==4)||matrix[3].some((value:number,index:number)=>Math.abs(value-(index===3?1:0))>1e-6))throw new Error('TCP pose requires a 4x4 rigid transform');
  const determinant=matrix[0][0]*(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])-matrix[0][1]*(matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0])+matrix[0][2]*(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0]);
  if(determinant<0)throw new Error('TCP pose must have a proper rotation');
  for(let i=0;i<3;i++)for(let j=0;j<3;j++)if(Math.abs(matrix.slice(0,3).reduce((sum:number,row:number[])=>sum+row[i]*row[j],0)-(i===j?1:0))>1e-4)throw new Error('TCP pose rotation must be orthonormal');
 };
 measure('plan_pose','Plan a complete world TCP pose using IK without acting or collision vetoes. Returns route, solver results and residuals.',{side,pose,path:{type:'string',enum:['auto','line','joint']}},async(a:any,e:any)=>{validPose(a.pose);const state_step=episode.read().step;return {...await robot.planPose(a.side,a.pose,a.path??'auto',e.signal),state_step};});
 motion('move_pose','Move to a complete world TCP pose, preserving its orientation. Returns actual errors, progress and state; no robot or payload collision screening runs.',{side,pose,path:{type:'string',enum:['auto','line','joint']},gripper:{type:'integer',enum:[-1,0,1]},max_steps:{type:'integer'},max_joint_step_rad:jointStep},async(a:any,e:any)=>{validPose(a.pose);if(a.max_steps!==undefined&&a.max_steps<1)throw new Error('max_steps must be positive');return robot.movePose(a.side,a.pose,a.max_steps??episode.read().step_limit-episode.read().step,a.path??'auto',a.gripper??0,e.signal,jointIncrement(a));});
 motion('move_trajectory','Execute explicit full TCP poses with straight segments. Returns the failed segment or actual progress; no automatic task recovery runs.',{side,poses:{type:'array',items:{type:'array',items:{type:'array',items:{type:'number'}}},required:true},max_steps:{type:'integer'},max_joint_step_rad:jointStep},async(a:any,e:any)=>{
  if(!a.poses.length||a.max_steps!==undefined&&a.max_steps<1)throw new Error('Nonempty poses and positive max_steps are required');
  a.poses.forEach(validPose);return robot.moveTrajectory(a.side,a.poses,a.max_steps??episode.read().step_limit-episode.read().step,e.signal,jointIncrement(a));
 });
 motion('move','Move TCP to world xyz metres. Euler degrees: no angles retains full current rotation; with any angle supplied, omissions use current Euler values except omitted roll=0 within one degree of vertical pitch. No automatic half-turn or alternate target is applied. line plans straight TCP waypoints, joint uses endpoint IK, auto authorizes either. No collision screening or lack-of-progress stop. Returns actual state, residuals, steps, waypoint progress and solver results; max_steps limits native actions. The other arm retains its last dispatched joint target, not a refreshed measured pose. motion_feedback reports measured/commanded joints and TCP displacement for both arms; tracking identifies the active waypoint.',{side,xyz,path:{type:'string',enum:['auto','line','joint']},pitch:{type:'number'},yaw:{type:'number'},roll:{type:'number'},max_steps:{type:'integer'},gripper:{type:'integer',enum:[-1,0,1]},max_joint_step_rad:jointStep},async(a:any,e:any)=>{
  if(a.xyz.length!==3||(a.max_steps!==undefined&&a.max_steps<1))throw new Error('xyz requires three values and max_steps must be positive');
  const start=robot.state()[a.side].tcp;
  const v=await robot.move(a.side,a.xyz,a.pitch,a.yaw,a.roll,a.max_steps??episode.read().step_limit-episode.read().step,e.signal,a.path??'auto',a.gripper??0,jointIncrement(a)),arm=v.state[a.side];
  return {arm:a.side,target_xyz:a.xyz,target_pose:v.target_pose,start_xyz:start,final_xyz:arm.tcp_pos,dist_to_target_m:v.position_error_m,orientation_error_deg:v.rotation_error_rad*180/Math.PI,
   pitch:arm.pitch,yaw:arm.yaw,roll:arm.roll,reached:v.reached,path:v.path,ik_error:v.plan.message,reason:v.reason,steps_used:v.steps,
   max_joint_error_rad:v.max_joint_error_rad,waypoints_completed:v.waypoints_completed,path_complete:v.path_complete,tracking:v.tracking,plan:v.plan,state:v.state,truncated:v.state.truncated};
 });
 motion('gripper','Command opening 1=open,0=closed for steps native actions, retaining the requested opening afterward. stable reports the last opening change below 0.004; it does not stop execution or establish object retention.',{side,opening:{type:'number',required:true},steps:{type:'integer',required:true}},(a:any,e:any)=>{if(a.opening<0||a.opening>1||a.steps<1)throw new Error('opening must be in [0,1] and steps positive');return robot.grip(a.side,a.opening,a.steps,e.signal);});
 motion('home','Return selected arms to initial joints, retaining opening commands. lift defaults false; explicit true requests a 6 cm straight rise per selected arm before the joint return. A failed rise returns feedback without attempting the return. No alternate rise, collision screening or hidden 25-action limit. Returns actual state, joint error, steps and lift solver results.',{side:{type:'string',enum:['left','right','both'],required:true},max_steps:{type:'integer'},lift:{type:'boolean'},max_joint_step_rad:jointStep},async(a:any,e:any)=>{
  if(a.max_steps!==undefined&&a.max_steps<1)throw new Error('max_steps must be positive');
  return robot.home(a.side,a.max_steps??episode.read().step_limit-episode.read().step,e.signal,a.lift??false,jointIncrement(a));
 });
 measure('kinematics','Query calibrated FK and geometric Jacobian for the named arm and explicit joints, or current public joints. Jacobian rows are TCP world translation and angular velocity; no hidden object state.',{side,joints:{type:'array',items:{type:'number'}}},(a:any)=>{
  const q=a.joints??robot.state()[a.side].joints;if(q.length!==6)throw new Error('Six joint values required');
  const frames=robot.kin.frames(a.side,q),tcp=robot.kin.tcp(a.side,q),origin=tcp.slice(0,3).map(row=>row[3]);
  const columns=frames.axes.map((axis,index)=>{const delta=origin.map((value,k)=>value-frames.origins[index][k]);return [axis[1]*delta[2]-axis[2]*delta[1],axis[2]*delta[0]-axis[0]*delta[2],axis[0]*delta[1]-axis[1]*delta[0],...axis];});
  return {arm:a.side,joints:q,tcp_pose:tcp,jacobian:Array.from({length:6},(_,index)=>columns.map(column=>column[index])),planning_joint_ranges_rad:robot.kin.chain.map(joint=>joint.limits),step:episode.read().step};
 });
 tool('finish','Stop the current policy with failure or stuck and summarize the observed obstacle. Native-confirmed success ends the episode automatically; no success status is accepted.',{status:{type:'string',enum:['failure','stuck'],required:true},summary:{type:'string',required:true}},(a:any)=>{episode.finish(a.status+': '+a.summary);return {finished:true};});
}
