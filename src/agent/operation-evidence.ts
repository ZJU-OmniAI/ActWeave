import type {FrameRef,Receipt} from '../episode/evidence.ts';

export type MotionSample={step:number;left:{tcp:number[];quaternion:number[]};right:{tcp:number[];quaternion:number[]}};
type Event={kind:string;start_step:number;end_step:number;arms:string[];receipt_id?:number;boundary_step?:number};

export function operationEvidence(input:{phase:number;start:number;end:number;anchors:FrameRef[];targets:FrameRef[];history:FrameRef[];extra:FrameRef[];cameras:string[];receipts:Receipt[];actions:any[];samples:MotionSample[];initialOpenings:Record<string,number>;inFlight:boolean}) {
 const chosen=new Map<string,FrameRef&{roles:string[]}>();
 const add=(step:number,cameras:string[],role:string)=>{
  for(const camera of cameras){
   const key=step+':'+camera,old=chosen.get(key);
   if(old){if(!old.roles.includes(role))old.roles.push(role);}
   else chosen.set(key,{step,camera,roles:[role]});
  }
 };
 for(const [refs,role] of [[input.anchors,'identity'],[input.targets,'target_reference'],[input.history,'historical_evidence'],[input.extra,'astra_selected']] as [FrameRef[],string][])
  for(const frame of refs)add(frame.step,[frame.camera],role);
 for(const [step,role] of [[input.phase,'phase_start'],[input.start,'window_start'],[input.end,input.inFlight?'current_in_flight':'current']] as [number,string][])add(step,input.cameras,role);
 const views=(arms:string[])=>input.cameras.filter(camera=>camera==='cam_head'||arms.some(arm=>camera===`cam_${arm}_wrist`));
 const events:Event[]=[];
 const openings={...input.initialOpenings};
 for(const action of input.actions){
  for(const side of ['left','right']){
   const opening=action.action[side+'_ee_joint_state'][0];
   if(opening!==openings[side]&&action.step>=input.start){
    const receipt=input.receipts.find(row=>row.kind==='motion'&&row.start_step<=action.step&&(row.end_step??input.end)>action.step&&(row.arguments.side===side||row.arguments.side==='both'||row.tool==='joint_action'));
    const end=receipt?.tool==='gripper'?(receipt.end_step??input.end):action.step+1;
    events.push({kind:'gripper_command_change',start_step:action.step,end_step:end,arms:[side],receipt_id:receipt?.receipt_id});
   }
   openings[side]=opening;
  }
 }
 for(const row of input.receipts){
  if(row.kind!=='motion')continue;
  const arms=row.arguments.side==='both'||row.tool==='joint_action'?['left','right']:row.arguments.side?[row.arguments.side]:[];
  if(row.status==='error'||row.result?.reached===false||row.end_step===null)events.push({kind:row.end_step===null?'in_flight_call':row.status==='error'?'call_error':'incomplete_motion',start_step:Math.max(input.start,row.start_step),end_step:Math.min(input.end,row.end_step??input.end),arms,receipt_id:row.receipt_id});
  if(row.tool==='wait'&&row.end_step!==null&&row.end_step>row.start_step){
   for(const fraction of [1/3,2/3])add(Math.round(Math.max(input.start,row.start_step)+(row.end_step-Math.max(input.start,row.start_step))*fraction),views([]),'wait_time_sample');
  }
 }
 for(const side of ['left','right']){
  let previous:{row:Receipt;delta:number[]}|null=null;
  for(const row of input.receipts){
   if(row.kind!=='motion'||row.tool==='gripper'||row.tool==='wait'||row.end_step===null||!(row.arguments.side===side||row.arguments.side==='both'||row.tool==='joint_action'))continue;
   const arm=row.result?.motion_feedback?.arms?.[side];
   if(!arm)continue;
   const delta=arm.tcp_end_m.map((value:number,k:number)=>value-arm.tcp_start_m[k]);
   if(!Math.hypot(...delta))continue;
   if(previous&&previous.delta.reduce((sum,value,k)=>sum+value*delta[k],0)<0){
    const boundary=previous.row.end_step!;
    events.push({kind:'tcp_direction_reversal',start_step:Math.max(input.start,boundary-1),boundary_step:boundary,end_step:Math.min(input.end,row.start_step+1),arms:[side],receipt_id:row.receipt_id});
   }
   previous={row,delta};
  }
 }
 for(const event of events){
  const times=event.kind==='gripper_command_change'?[event.start_step,Math.round((event.start_step+event.end_step)/2),Math.min(input.end,event.end_step+1)]:[event.start_step,event.end_step];
  if(event.boundary_step!==undefined)times.push(event.boundary_step);
  for(const step of times)add(step,views(event.arms),event.kind);
 }
 for(const side of ['left','right'] as const){
  const moving=input.receipts.some(row=>row.kind==='motion'&&row.tool!=='gripper'&&row.tool!=='wait'&&(row.arguments.side===side||row.arguments.side==='both'||row.tool==='joint_action'));
  if(!moving||input.samples.length<2)continue;
  const distances=[0],angles=[0];
  for(let i=1;i<input.samples.length;i++){
   const a=input.samples[i-1][side],b=input.samples[i][side];
   distances.push(distances.at(-1)!+Math.hypot(...b.tcp.map((v,k)=>v-a.tcp[k])));
   const dot=Math.abs(a.quaternion.reduce((sum,v,k)=>sum+v*b.quaternion[k],0));
   const scale=Math.hypot(...a.quaternion)*Math.hypot(...b.quaternion);
   angles.push(angles.at(-1)!+2*Math.acos(Math.min(1,dot/scale)));
  }
  for(const [progress,role] of [[distances,'translation_progress'],[angles,'rotation_progress']] as [number[],string][]){
   const total=progress.at(-1)!;
   if(total===0)continue;
   for(const fraction of [1/3,2/3]){
    const index=progress.findIndex(value=>value>=fraction*total);
    add(input.samples[index].step,views([side]),role);
   }
  }
 }
 const selected=[...chosen.values()].sort((a,b)=>a.step-b.step||a.camera.localeCompare(b.camera));
 const times=[...new Set(selected.filter(f=>f.step>=input.start&&f.step<=input.end).map(f=>f.step))].sort((a,b)=>a-b);
 const gaps=times.slice(1).flatMap((step,i)=>step>times[i]+1?[[times[i]+1,step]]:[]);
 return {frames:selected.map(({roles,...frame})=>frame),coverage:{
  window_start:input.start,window_end:input.end,phase_start:input.phase,in_flight_action:input.inFlight,
  selected_frames:selected,events,unsent_step_intervals:gaps,
  sampling:'Representative observations, not a continuous video. Intervals are [start,end). Events label commands or execution outcomes, not grasp/contact success. Motion samples use cumulative measured TCP translation and rotation over this observation window, not every solver waypoint. No image/token budget filtering is applied.',
 }};
}
