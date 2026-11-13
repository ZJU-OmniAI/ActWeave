import {defineTool} from '@deepseek-ai/dsh-tools';
import {toolValue} from '../tools/value.ts';
import type {EpisodeEvidence,Association} from './evidence.ts';

export type ExecutionPolicy={kind?:'motion'|'measurement';rawMotion?:boolean;sourceSteps?:(args:any,start:number)=>number[]};
export type Register=(name:string,description:string,parameters:any,execute:any,render?:any,policy?:ExecutionPolicy)=>void;

export class CapabilityExecution {
 constructor(private registry:any,private evidence:EpisodeEvidence,private association:()=>Association) {}
 register:Register=(name,description,parameters,execute,render=(value:any)=>[{type:'text',text:JSON.stringify(value)}],policy={})=>{
  this.registry.register(defineTool({name,description,parameters,output:{schema:{type:'json'},render:(_args:any,value:any)=>render(value)},execute:async(args:any,exec:any)=>{
   const robot=this.evidence.robot,episode=this.evidence.episode,startStep=this.evidence.step;
   const before=policy.kind==='motion'?this.evidence.state():null;
   const receipt=policy.kind?this.evidence.begin(policy.kind,name,args,exec.callId,policy.sourceSteps?.(args,startStep)??[startStep],this.association()):null;
   try {
    const value=await execute(args,exec);
    const result=toolValue(before&&!policy.rawMotion?{...value,start_step:startStep,end_step:this.evidence.step,motion_feedback:robot.motionFeedback(before,args.max_steps,value.reason??null)}:value);
    if(receipt)this.evidence.returned(receipt,result);
    return result;
   } catch(error) {
    if(before){
     const message=JSON.stringify({error:String(error),start_step:startStep,end_step:episode.read().step,state:this.evidence.state(),motion_feedback:robot.motionFeedback(before,args.max_steps,null),note:'Physical effects remain; end_step is the latest observed step, not proof that a cancelled action has settled'});
     if(receipt)this.evidence.failed(receipt,message);
     throw new Error(message,{cause:error});
    }
    if(receipt)this.evidence.failed(receipt,String(error));
    throw error;
   }
  }}));
 };
}
