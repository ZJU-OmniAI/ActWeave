import type {Receipt} from '../episode/evidence.ts';
import type {TransitionDescription} from './transitions.ts';

const pick=(value:any,keys:string[])=>Object.fromEntries(keys.filter(key=>value[key]!==undefined).map(key=>[key,value[key]]));

function receiptFacts(receipt:Receipt) {
 const {receipt_id,kind,tool,status,start_step,end_step,evidence_steps}=receipt;
 const facts:any={receipt_id,kind,tool,status,start_step,end_step,evidence_steps};
 facts.request=pick(receipt.arguments,['side','camera','xyz','pose','pitch','yaw','roll','path','gripper','opening','steps','max_steps','max_joint_step_rad','row','col','z','support_z','prompt','view_prompts','region_index']);
 if(receipt.result!==undefined){
  const result=receipt.result;
  facts.result=pick(result,['found','camera','step','state_step','center_xyz','xyz','top_z','height_m','length_m','width_m','grasp_yaw','geometry','surface','note','views','arm','target_xyz','final_xyz','reached','reason','steps','steps_used','position_error_m','dist_to_target_m','orientation_error_deg','rotation_error_rad','path_complete','waypoints_completed','commanded','opening','holding','stable']);
  if(result.ik)facts.result.ik=pick(result.ik,['reached','message','position_error_m','rotation_error_rad']);
  if(result.motion_feedback){
   facts.execution=pick(result.motion_feedback,['executed_steps','caller_max_steps','native_steps_remaining','limit_kind']);
   facts.execution.arms=Object.fromEntries(Object.entries(result.motion_feedback.arms).map(([side,arm]:[string,any])=>[side,pick(arm,['tcp_start_m','tcp_end_m'])]));
  }
 }
 if(receipt.error!==undefined){
  const error=kind==='motion'?JSON.parse(receipt.error):{error:receipt.error};
  facts.error=error.error;
  if(error.note!==undefined)facts.error_note=error.note;
 }
 return facts;
}

export function watchContext(
 record:{id:string;revision:number;phase_step:number;description:TransitionDescription},
 roles:{identity_anchors:any;phase_step:number;recent_step:number;current_step:number},
 executions:Receipt[],basis:Receipt[],previousJudgment:any,packet?:{coverage:any;historical_evidence:any[]},
) {
 const {watch,anchors,target_anchors,depends_on,basis_receipts,decision,...intent}=record.description;
 const receipts=[...new Map([...executions,...basis].map(row=>[row.receipt_id,row])).values()];
 return JSON.stringify({
  source:'Astra-authored hypotheses, not observed truth',
  transition:{id:record.id,revision:record.revision,description:intent},
  evidence_roles:roles,
  receipt_evidence:{
   scope:'Fact projection, not complete receipts: targets, outcome, actual step ranges, TCP endpoints and scalar geometry. Solver plans, per-call joint arrays, masks, pixel clouds and camera matrices are not included here. Selected command steps are preserved separately in state.actions; selected-frame robot states are in state.public_states. Full receipts, snapshots and original errors remain available to Astra through transition_read and transition_observe. A returned call or holding heuristic does not certify an object effect.',
   execution_receipt_ids:executions.map(row=>row.receipt_id),basis_receipt_ids:basis.map(row=>row.receipt_id),receipts:receipts.map(receiptFacts),
  },
  previous_judgment:previousJudgment,
  ...(packet??{}),
  observer_instructions:'Compare the named instance and reference across phase, recent and current evidence. Identity anchors identify objects; old images and earlier answers do not prove current retention or support. Camera motion is not object motion. Caller goals and expectations are hypotheses. Distinguish changed, maintained, contradicted and unobservable relations using the caller choices; do not infer task completion or an action permit. Each question is independent. A missing alternative or ambiguous identity calls for unknown. Execute no actions and prescribe no phase transition.',
 });
}
