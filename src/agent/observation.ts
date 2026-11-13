import type {EpisodeEvidence,Association} from '../episode/evidence.ts';
import type {AgentRuntime} from '../runtime/agent.ts';
import {analyzeJev,type JevQuery} from './jev.ts';
import type {TransitionStore} from './transitions.ts';
import {AstraPresentation} from './presentation.ts';

export class ObservationCoordinator {
 constructor(private evidence:EpisodeEvidence,private transitions:TransitionStore,private presentation:AstraPresentation,private config:AgentRuntime) {}
 private async consult(query:JevQuery,exec:any,association:Association) {
  const record=this.evidence.beginConsultation(query,exec.callId,association);
  if(!this.config.jev){record.status='unavailable';record.error='Jev is disabled in this profile; no consultation was attempted.';return {record};}
  try {
   record.result=await analyzeJev(this.evidence,this.config.jev,query,this.config.deadlineAt,AbortSignal.any([this.evidence.episode.stopped.signal,exec.signal]),exec.callId);
   record.status='returned';
  }catch(error){record.status='error';record.error=String(error);return {record,error};}
  return {record};
 }
 async analyze(query:JevQuery,exec:any) {
  const {record,error}=await this.consult(query,exec,this.transitions.association());
  if(error!==undefined)throw error;
  if(record.status!=='returned')throw new Error(record.error);
  return record.result;
 }
 async observe(id:string,frames:any,longContext:boolean|undefined,exec:any) {
  const prepared=await this.transitions.prepare(id,frames,longContext),snapshots=this.evidence.snapshots(prepared.frames);
  const basis=prepared.basis;
  const images=await this.presentation.pictures(snapshots);
  let observation:any,reused=false;
  if(prepared.cached){observation=prepared.cached;reused=true;}
  else{
   const association={transition_id:prepared.record.id,revision:prepared.record.revision};
   let consultation;
   if(!prepared.query){
    consultation=this.evidence.beginConsultation(null,exec.callId,association);
    consultation.status='unavailable';consultation.error='No watch is registered; Astra can revise the description or use independent tools.';
   }else{
    consultation=(await this.consult(prepared.query,exec,association)).record;
   }
   observation=this.transitions.observe(prepared,consultation.consultation_id);
  }
  return {transition:prepared.record,receipts:prepared.receipts,window_receipts:prepared.windowReceipts,coverage:prepared.coverage,historical_evidence:prepared.history,basis_evidence:basis,observation,reused,snapshots,images,
   scope:{receipt_revision:prepared.record.revision,observation_step:prepared.current,returned_step:this.evidence.step,in_flight_action:Boolean(this.evidence.episode.pending)},
   instruction:'Compare original execution feedback and visual advice with your hypothesis. You decide whether to continue, revise, gather evidence or stop. Returned calls do not certify intended effects; old answers do not prove current state. No next action has been selected.'};
 }
}
