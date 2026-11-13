import type {JevQuery} from './jev.ts';

import {publicCameras,cameraNames,frameRefs,type FrameRef,type EpisodeEvidence,type Association} from '../episode/evidence.ts';
import {validateQuestions} from './jev.ts';
import {watchContext} from './jev-context.ts';
import {operationEvidence} from './operation-evidence.ts';

export type TransitionDescription={
 objects:Record<string,string>;basis:string;goal_delta:string;expected_delta:string;preserve:string[];decision:string;
 anchors:FrameRef[];target_anchors?:FrameRef[];depends_on?:{id:string;revision?:number}[];basis_receipts?:number[];watch?:{questions:JevQuery['questions'];cameras?:string[];long_context?:boolean};
};
type Revision={id:string;revision:number;recorded_step:number;phase_step:number;description:TransitionDescription};
type Observation={observation_id:number;transition_id:string;revision:number;step:number;frames:FrameRef[];receipt_ids:number[];consultation_id:number;coverage:ReturnType<typeof operationEvidence>['coverage'];key:string};

function object(value:any,label:string) {
 if(!value||typeof value!=='object'||Array.isArray(value))throw new Error(label+' must be an object');
}
function nonempty(value:any,label:string) {
 if(typeof value!=='string'||!value.trim())throw new Error(label+' must be a nonempty string');
}
function validateDescription(value:any,current:number):TransitionDescription {
 object(value,'description');object(value.objects,'objects');
 if(!Object.keys(value.objects).length)throw new Error('objects must identify at least one instance or reference');
 for(const [name,description] of Object.entries(value.objects)){nonempty(name,'object name');nonempty(description,'object description');}
 for(const field of ['basis','goal_delta','expected_delta','decision'])nonempty(value[field],field);
 if(!Array.isArray(value.preserve)||value.preserve.some((item:any)=>typeof item!=='string'||!item.trim()))throw new Error('preserve must be an array of nonempty descriptions');
 const anchors=frameRefs(value.anchors,current);
 if(!anchors.length)throw new Error('anchors must include an identity reference');
 if(value.target_anchors!==undefined)frameRefs(value.target_anchors,current);
 if(value.depends_on!==undefined){
  if(!Array.isArray(value.depends_on))throw new Error('depends_on must be an array of transition references');
  for(const dependency of value.depends_on){object(dependency,'dependency');nonempty(dependency.id,'dependency.id');if(dependency.revision!==undefined&&(!Number.isInteger(dependency.revision)||dependency.revision<1))throw new Error('dependency revision must be positive');}
 }
 if(value.watch!==undefined){
  object(value.watch,'watch');validateQuestions(value.watch.questions);if(value.watch.cameras!==undefined)cameraNames(value.watch.cameras);
  if(value.watch.long_context!==undefined&&typeof value.watch.long_context!=='boolean')throw new Error('long_context must be boolean');
 }
 return structuredClone({...value,anchors});
}
function cursor(value:any,label:string) {
 if(!Number.isInteger(value)||value<0)throw new Error(label+' must be a nonnegative integer');
}

function historicalAnswers(answers:any,question?:string) {
 return Object.fromEntries(Object.entries(answers).filter(([id])=>question===undefined||id===question).map(([id,answer]:[string,any])=>[id,{
  choice:answer.choice,confidence:answer.confidence,probabilities:answer.probabilities,unresolved:answer.unresolved,
  ...(answer.observation_gap?{observation_gap:{choice:answer.observation_gap.choice,reason:answer.observation_gap.reason,next_observation:answer.observation_gap.next_observation}}:{}),
 }]));
}

export class TransitionStore {
 private records=new Map<string,Revision[]>();
 private observations:Observation[]=[];
 private active:string|null=null;
 constructor(private evidence:EpisodeEvidence) {}
 record(id:string,description?:TransitionDescription,active?:boolean) {
  nonempty(id,'id');
  if(active!==undefined&&typeof active!=='boolean')throw new Error('active must be boolean');
  if(description!==undefined){
   const current=this.evidence.step,validated=validateDescription(description,current);
   for(const frame of validated.anchors)this.evidence.raw(frame.step);
   if(validated.basis_receipts!==undefined){
    if(!Array.isArray(validated.basis_receipts))throw new Error('basis_receipts must be receipt ids');
    for(const receipt of validated.basis_receipts)this.evidence.receipt(receipt);
   }
   const history=this.records.get(id)??[];
   validated.target_anchors=frameRefs(validated.target_anchors??history.at(-1)?.description.target_anchors??validated.anchors,current);
   validated.depends_on=validated.depends_on?.map(dependency=>{const source=this.revision(dependency.id,dependency.revision);return {id:source.id,revision:source.revision};});
   history.push({id,revision:history.length+1,recorded_step:current,phase_step:current,description:validated});
   this.records.set(id,history);
  }
  const record=this.revision(id);
  if(active===true)this.active=id;
  else if(active===false&&this.active===id)this.active=null;
  return {record:structuredClone(record),active:this.active,source:'Astra-authored description and assumptions; no physical state is certified'};
 }
 private revision(id:string,version?:number) {
  const history=this.records.get(id);
  if(!history)throw new Error('Unknown transition: '+id);
  if(version!==undefined&&(!Number.isInteger(version)||version<1||version>history.length))throw new Error('Unknown transition revision');
  return history[(version??history.length)-1];
 }
 read(id?:string,version?:number,afterReceipt=0,afterObservation=0) {
  cursor(afterReceipt,'after_receipt');cursor(afterObservation,'after_observation');
  if(id===undefined)return {evidence:this.evidence.read(afterReceipt,afterObservation),active:this.active,transitions:[...this.records.values()].map(history=>({id:history[0].id,revision:history.length,recorded_step:history.at(-1)!.recorded_step}))};
  const record=this.revision(id,version);
  return structuredClone({record,active:this.active,sampled_step:this.evidence.step,revisions:this.records.get(id)!.map(row=>({revision:row.revision,recorded_step:row.recorded_step})),
   receipts:this.evidence.executions({transition_id:id,revision:record.revision},afterReceipt),
   observations:this.observations.filter(row=>row.transition_id===id&&row.revision===record.revision&&row.observation_id>afterObservation).map(row=>this.observation(row)),
   scope:{revision:record.revision,after_receipt:afterReceipt,after_observation:afterObservation},source:'Descriptions are Astra assertions; Jev outputs are advice. Recorded steps are not proof of continued validity.'});
 }
 association():Association {return this.active===null?null:{transition_id:this.active,revision:this.revision(this.active).revision};}
 private observation(row:Observation) {
  const {key,...record}=row,consultation=this.evidence.consultation(row.consultation_id);
  return {...record,query:consultation.query,consultation:{status:consultation.status,...(consultation.result===undefined?{}:{result:consultation.result}),...(consultation.error===undefined?{}:{error:consultation.error})}};
 }
 private historicalEvidence(record:Revision,current:number) {
  const entries:any[]=[],frames:FrameRef[]=[];
  const add=(source:Revision,role:string,observations:Observation[],question?:string)=>{
   const selected=[...new Map(observations.map(row=>[row.observation_id,row])).values()];
   const lastExecution=role==='declared_dependency'?this.evidence.executions({transition_id:source.id,revision:source.revision}).filter(row=>row.end_step!==null).at(-1):undefined;
   const executionFrames=lastExecution?Object.keys(this.evidence.raw(lastExecution.end_step!).images).map(camera=>({step:lastExecution.end_step!,camera})):[];
   const evidenceFrames=frameRefs([...source.description.anchors,...(source.description.target_anchors??[]),...executionFrames,...selected.flatMap(row=>row.frames.filter(frame=>frame.step>=(row.coverage?.window_start??source.phase_step)&&frame.step<=row.step))],current);
   entries.push({role,transition_id:source.id,revision:source.revision,objects:source.description.objects,goal_delta:source.description.goal_delta,basis:source.description.basis,
    basis_receipt_ids:[...new Set([...(source.description.basis_receipts??[]),...(lastExecution?[lastExecution.receipt_id]:[])])],last_execution:lastExecution?{receipt_id:lastExecution.receipt_id,step:lastExecution.end_step,status:lastExecution.status}:null,question_id:question,frames:evidenceFrames,judgments:selected.map(row=>{const observation=this.observation(row);return {observation_id:row.observation_id,step:row.step,questions:question?{[question]:observation.query.questions[question]}:observation.query.questions,answers:historicalAnswers(observation.consultation.result.answers,question)};}),
    scope:'Historical Astra hypotheses and dated Jev advice with their source frames included in this request. Not a current fact or automatic confirmation.'});
   frames.push(...evidenceFrames);
  };
  for(const dependency of record.description.depends_on??[]){
   const source=this.revision(dependency.id,dependency.revision);
   const observations=this.observations.filter(row=>row.transition_id===source.id&&row.revision===source.revision&&this.evidence.consultation(row.consultation_id).status==='returned');
   add(source,'declared_dependency',observations.length?[observations[0],observations.at(-1)!]:[]);
  }
  const observations=this.observations.filter(row=>row.transition_id===record.id&&row.revision===record.revision&&row.step<current&&this.evidence.consultation(row.consultation_id).status==='returned');
  if(observations.length)add(record,'previous_observation',[observations.at(-1)!]);
  for(const question of Object.keys(record.description.watch?.questions??{})){
   const unresolved:Observation[]=[];
   for(const row of observations){
    const answer=this.evidence.consultation(row.consultation_id).result.answers[question];
    if(!answer)continue;
    if(answer.choice==='unknown'||answer.unresolved?.length)unresolved.push(row);else unresolved.length=0;
   }
   if(unresolved.length)add(record,'unresolved_question',[unresolved[0],unresolved.at(-1)!],question);
  }
  return {entries,frames:frameRefs(frames,current)};
 }
 async prepare(id:string,extraFrames:any=[],longContext?:boolean) {
  if(longContext!==undefined&&typeof longContext!=='boolean')throw new Error('long_context must be boolean');
  const record=structuredClone(this.revision(id)),current=this.evidence.step,watch=record.description.watch,inFlight=Boolean(this.evidence.episode.pending);
  const previous=[...this.observations].reverse().find(row=>row.transition_id===id&&row.revision===record.revision&&this.evidence.consultation(row.consultation_id).status==='returned'&&row.step<current);
  const recent=previous?.step??record.phase_step;
  const receipts=structuredClone(this.evidence.executions({transition_id:id,revision:record.revision}));
  const windowReceipts=structuredClone(this.evidence.executions().filter(row=>row.start_step<=current&&(row.end_step===null||row.end_step>recent||row.start_step>=recent)));
  const historical=this.historicalEvidence(record,current);
  const basis=[...new Set([...(record.description.basis_receipts??[]),...historical.entries.flatMap(entry=>entry.basis_receipt_ids)])].map(id=>this.evidence.receipt(id));
  const openings=this.evidence.episode.stateAt(Math.max(0,recent-1));
  const packet=operationEvidence({phase:record.phase_step,start:recent,end:current,anchors:record.description.anchors,targets:record.description.target_anchors??[],history:historical.frames,
   extra:frameRefs(extraFrames,current),cameras:publicCameras.filter(camera=>camera in this.evidence.raw().images),receipts:windowReceipts,
   actions:await this.evidence.commands(Math.max(0,recent-1),current),samples:this.evidence.motionSamples(recent,current),
   initialOpenings:{left:openings.left_ee_joint_state[0],right:openings.right_ee_joint_state[0]},inFlight});
  const frames=packet.frames;
  const query:JevQuery|null=watch?{
   questions:watch.questions,frames,action_range:[recent,current],decision:record.description.decision,long_context:longContext??watch.long_context??false,
   context:watchContext(record,
    {identity_anchors:record.description.anchors,phase_step:record.phase_step,recent_step:recent,current_step:current},
    windowReceipts,basis,
    previous?{observation_id:previous.observation_id,step:previous.step,answers:historicalAnswers(this.evidence.consultation(previous.consultation_id).result.answers),source:'Historical Jev advice, not current facts'}:null,
    {coverage:packet.coverage,historical_evidence:historical.entries})
  }:null;
  const key=JSON.stringify({revision:record.revision,current,frames,receipts:windowReceipts.map(row=>[row.receipt_id,row.status,row.end_step]),history:historical.entries,long_context:query?.long_context});
  const cached=[...this.observations].reverse().find(row=>row.transition_id===id&&row.key===key&&this.evidence.consultation(row.consultation_id).status==='returned');
  return {record,current,recent,frames,receipts,windowReceipts,basis,coverage:packet.coverage,history:historical.entries,query,key,cached:cached?this.observation(cached):null};
 }
 observe(prepared:Awaited<ReturnType<TransitionStore['prepare']>>,consultationId:number) {
  const observation:Observation={observation_id:this.observations.length+1,transition_id:prepared.record.id,revision:prepared.record.revision,step:prepared.current,
   frames:prepared.frames,receipt_ids:prepared.windowReceipts.map(row=>row.receipt_id),consultation_id:consultationId,coverage:prepared.coverage,key:prepared.key};
  this.observations.push(observation);return this.observation(observation);
 }
 context() {
  if(!this.records.size)return null;
  const latest=this.active===null?null:this.revision(this.active);
  const observation=latest?[...this.observations].reverse().find(row=>row.transition_id===latest.id&&row.revision===latest.revision):null;
  return {active:latest?{id:latest.id,revision:latest.revision,description:latest.description,phase_step:latest.phase_step}:null,
   sampled_step:this.evidence.step,last_observation:observation?{id:observation.observation_id,step:observation.step,status:this.evidence.consultation(observation.consultation_id).status,error:this.evidence.consultation(observation.consultation_id).error}:null,
   instruction:'Astra decides whether to continue, revise, observe or stop. Use transition_read for original records. Old advice is not current state; no automatic observation or action gate is active.'};
 }
 export() {return {active:this.active,records:[...this.records.values()],observations:this.observations.map(({key,...row})=>row)};}
}
