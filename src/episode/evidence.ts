import {readFile} from 'node:fs/promises';
import path from 'node:path';
import type {Robot} from '../robot/control.ts';
import type {Vision} from '../vision/vision.ts';
import {apply,quatPose} from '../robot/math.ts';
import type {MotionSample} from '../agent/operation-evidence.ts';
import type {JevQuery} from '../agent/jev.ts';

export const publicCameras=['cam_head','cam_left_wrist','cam_right_wrist'];
export type FrameRef={step:number;camera:string};
export type Association={transition_id:string;revision:number}|null;
export type Receipt={receipt_id:number;call_id:string;program_call_id:string|null;kind:'motion'|'measurement';tool:string;arguments:any;start_step:number;end_step:number|null;evidence_steps:number[];transition_id:string|null;revision:number|null;status:'started'|'returned'|'error';result?:any;error?:string};
export type Consultation={consultation_id:number;call_id:string;query:JevQuery|null;step:number;association:Association;status:'started'|'returned'|'error'|'unavailable';result?:any;error?:string};

export function cameraNames(value:any) {
 if(!Array.isArray(value)||!value.length||value.some(name=>!publicCameras.includes(name))||new Set(value).size!==value.length)throw new Error('cameras must contain distinct public camera names');
 return value as string[];
}
export function frameRefs(value:any,current:number) {
 if(!Array.isArray(value))throw new Error('frames must be an array');
 const frames=value.map((frame:any)=>{
  if(!frame||!publicCameras.includes(frame.camera)||!Number.isInteger(frame.step)||frame.step< -1||frame.step>current)throw new Error('frames require public cameras and existing steps, or -1 for current');
  return {camera:frame.camera,step:frame.step===-1?current:frame.step};
 });
 return [...new Map(frames.map(frame=>[`${frame.step}:${frame.camera}`,frame])).values()];
}

export class EpisodeEvidence {
 private receipts:Receipt[]=[];
 private consultations:Consultation[]=[];
 private programs=new Map<string,any>();
 private parents=new Map<string,string>();
 private decisionReceipt=0;
 private decisionConsultation=0;
 constructor(readonly robot:Robot,readonly vision:Vision) {}
 get episode(){return this.robot.episode;}
 get step(){return this.episode.read().step;}
 raw(step=-1){return this.episode.read(step);}
 state(step=-1){return this.robot.publicState(step);}
 calibration(camera:string,step=-1) {
  const observation=this.raw(step),view=this.vision.camera(camera,observation.step);
  return {camera,step:observation.step,shape:[view.height,view.width,3],intrinsic_matrix:[[view.fx,0,view.cx],[0,view.fy,view.cy],[0,0,1]],extrinsic_matrix:view.transform};
 }
 image(camera:string,step=-1) {
  const observation=this.raw(step);
  return {camera,step:observation.step,png_base64:observation.images[camera]};
 }
 snapshot(step:number,names:string[]) {
  if(!Number.isInteger(step)||step< -1||step>this.step)throw new Error('step must be -1 or an existing public step');
  const observation=this.raw(step),{ended,...state}=this.state(observation.step);
  const requestedOpenings=Object.fromEntries(['left','right'].map(side=>{const {requested_opening,...measured}=state[side];state[side]=measured;return [side,requested_opening];}));
  return {step:observation.step,state,runtime:{episode_ended:ended,requested_openings:requestedOpenings,captured_at_step:this.step},
   cameras:names.map(camera=>({...this.calibration(camera,observation.step),...this.image(camera,observation.step)}))};
 }
 snapshots(frames:FrameRef[]) {
  return [...new Set(frames.map(frame=>frame.step))].map(step=>this.snapshot(step,frames.filter(frame=>frame.step===step).map(frame=>frame.camera)));
 }
 motionSamples(start:number,end:number):MotionSample[] {
  return Array.from({length:end-start+1},(_,index)=>{
   const step=start+index,state=this.episode.stateAt(step);
   const arm=(side:string)=>{const pose=state[side+'_ee_pose'];return {tcp:apply(quatPose(pose),[this.robot.kin.config.robot.gripper_bias,0,0]),quaternion:pose.slice(3)};};
   return {step,left:arm('left'),right:arm('right')};
  });
 }
 async commands(start:number,end:number) {
  if(start===end)return [];
  const lines=await readFile(path.join(this.episode.directory,'actions.jsonl'),'utf8');
  return lines.trimEnd().split('\n').map(line=>JSON.parse(line)).filter(row=>row.step>=start&&row.step<end);
 }
 begin(kind:Receipt['kind'],tool:string,args:any,callId:string,sources:number[],association:Association) {
  const receipt:Receipt={receipt_id:this.receipts.length+1,call_id:callId,program_call_id:this.parents.get(callId)??null,kind,tool,arguments:structuredClone(args),
   start_step:this.step,end_step:null,evidence_steps:sources,transition_id:association?.transition_id??null,revision:association?.revision??null,status:'started'};
  this.receipts.push(receipt);return receipt;
 }
 returned(receipt:Receipt,result:any) {receipt.end_step=this.step;receipt.status='returned';receipt.result=result;}
 failed(receipt:Receipt,error:string) {receipt.end_step=this.step;receipt.status='error';receipt.error=error;}
 executions(association?:Association,after=0) {
  return this.receipts.filter(row=>{
   if(row.receipt_id<=after)return false;
   if(association===undefined)return true;
   return association===null?row.transition_id===null:row.transition_id===association.transition_id&&row.revision===association.revision;
  });
 }
 receipt(id:number) {
  if(!Number.isInteger(id)||id<1||id>this.receipts.length)throw new Error('Unknown evidence receipt: '+id);
  return this.receipts[id-1];
 }
 beginConsultation(query:JevQuery|null,callId:string,association:Association) {
  const record:Consultation={consultation_id:this.consultations.length+1,call_id:callId,query:structuredClone(query),step:this.step,association,status:'started'};
  this.consultations.push(record);return record;
 }
 consultation(id:number) {
  if(!Number.isInteger(id)||id<1||id>this.consultations.length)throw new Error('Unknown consultation: '+id);
  return this.consultations[id-1];
 }
 sessionEvent(event:any) {
  const data=event.data;
  if(event.type==='assistant/message'){this.decisionReceipt=this.receipts.length;this.decisionConsultation=this.consultations.length;}
  if(event.type==='tool/call'&&data.name==='run_code')this.programs.set(data.callId,{call_id:data.callId,arguments:data.arguments,start_event_seq:event.seq,start_step:this.step,status:'started'});
  if(event.type==='tool/result') {
   const program=this.programs.get(data.message.toolCallId);
   if(program)Object.assign(program,{result_event_seq:event.seq,end_step:this.step,status:data.message.isError?'error':'returned'});
  }
  if(event.type==='tool/ptc-dispatch-start'||event.type==='tool/ptc-dispatch') {
   this.parents.set(data.subCallId,data.rootCallId);
   const receipt=this.receipts.find(row=>row.call_id===data.subCallId);
   if(receipt)receipt.program_call_id=data.rootCallId;
  }
 }
 context() {
  const receipt=this.receipts.at(-1),consultation=this.consultations.at(-1);
  const executions=this.receipts.filter(row=>row.receipt_id>this.decisionReceipt).map(row=>{
   const {result,...reference}=row;
   const fields=row.kind==='motion'?['reached','reason','steps','steps_used','target_xyz','target_pose','final_xyz','position_error_m','dist_to_target_m','orientation_error_deg','rotation_error_rad','path_complete','waypoints_completed','tracking','motion_feedback','state']:['found','step','state_step','center_xyz','xyz','top_z','height_m','grasp_yaw','geometry','views','note','reason','ik'];
   return {...reference,result:result===undefined?undefined:row.tool==='joint_action'?result:Object.fromEntries(fields.filter(key=>result[key]!==undefined).map(key=>[key,result[key]])),
    arguments:row.kind==='motion'?row.arguments:undefined,full_result:{tool:'transition_read',after_receipt:row.receipt_id-1}};
  });
  return {sampled_step:this.step,in_flight_action:Boolean(this.episode.pending),
   decision_scope:{after_receipt:this.decisionReceipt,after_consultation:this.decisionConsultation},executions,
   consultations:this.consultations.filter(row=>row.consultation_id>this.decisionConsultation),
   last_execution:receipt?{receipt_id:receipt.receipt_id,tool:receipt.tool,status:receipt.status,start_step:receipt.start_step,end_step:receipt.end_step,evidence_steps:receipt.evidence_steps,program_call_id:receipt.program_call_id}:null,
   last_consultation:consultation?{consultation_id:consultation.consultation_id,step:consultation.step,status:consultation.status}:null,
   instruction:'These are original call facts since your previous reply, including intermediate calls. Compare actual execution with the assumptions of dependent actions. Object effects require your interpretation of measurements and images. Full original results remain in tool history and transition_read; no physical decision or next action has been selected.'};
 }
 read(afterReceipt=0,afterConsultation=0) {
  return structuredClone({sampled_step:this.step,programs:[...this.programs.values()],receipts:this.executions(undefined,afterReceipt),consultations:this.consultations.filter(row=>row.consultation_id>afterConsultation),
   scope:{after_receipt:afterReceipt,after_consultation:afterConsultation}});
 }
 export(){return this.read();}
}
