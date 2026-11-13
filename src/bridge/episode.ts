import { mkdir,writeFile,appendFile } from 'node:fs/promises';
import path from 'node:path';
import { readFileSync } from 'node:fs';
export class Episode {
 observation:any;
 private states=new Map<number,any>();
 pending?: {action:any; resolve:(value:any)=>void; reject:(error:Error)=>void};
 lastAction?:Record<string,number[]>;
 ended=false;
 reason='';
 stopped=new AbortController();
 constructor(public directory:string) {}
 async update(observation:any) {
  this.observation=observation;
  this.states.set(observation.step,structuredClone(observation.state));
  await mkdir(this.directory,{recursive:true});
  const folder=path.join(this.directory,String(observation.step).padStart(4,'0'));
  await mkdir(folder,{recursive:true});
  for(const [name,image] of Object.entries(observation.images)) await writeFile(path.join(folder,name+'.png'),Buffer.from(image as string,'base64'));
  await writeFile(path.join(folder,'observation.json'),JSON.stringify({...observation,images:Object.keys(observation.images)}));
  const pending=this.pending;this.pending=undefined;pending?.resolve(observation);
 }
 read(step=-1) {
  if(step<0||step===this.observation?.step){if(!this.observation)throw new Error('Observation unavailable');return this.observation;}
  const folder=path.join(this.directory,String(step).padStart(4,'0'));
  const observation=JSON.parse(readFileSync(path.join(folder,'observation.json'),'utf8'));
  return {...observation,images:Object.fromEntries(observation.images.map((name:string)=>[name,readFileSync(path.join(folder,name+'.png')).toString('base64')]))};
 }
 stateAt(step:number) {
  if(this.states.has(step))return this.states.get(step);
  const folder=path.join(this.directory,String(step).padStart(4,'0'));
  return JSON.parse(readFileSync(path.join(folder,'observation.json'),'utf8')).state;
 }
 async action(action:any,signal?:AbortSignal) {
  if(this.ended)throw new Error('Episode ended: '+this.reason);
  if(this.pending)throw new Error('An action is already pending');
  await appendFile(path.join(this.directory,'actions.jsonl'),JSON.stringify({step:this.observation.step,action})+'\n');
  if(this.ended)throw new Error('Episode ended: '+this.reason);
  return new Promise<any>((resolve,reject)=>{
   const cancelled=()=>new Error('Action call cancelled; physical effects remain; read the current observation before continuing');
   if(signal?.aborted){reject(cancelled());return;}
   let interrupted=false;
   const abort=()=>{
    interrupted=true;
    if(pending.action!==null){this.pending=undefined;pending.reject(cancelled());}
   };
   const pending={action,resolve:(value:any)=>{signal?.removeEventListener('abort',abort);interrupted?reject(cancelled()):resolve(value);},reject:(error:Error)=>{signal?.removeEventListener('abort',abort);reject(error);}};
   this.pending=pending;
   signal?.addEventListener('abort',abort,{once:true});
  });
 }
 take() { if(this.ended)return {status:'finished',reason:this.reason};if(!this.pending?.action)return {status:'waiting'};const action=this.pending.action;this.pending.action=null;this.lastAction=structuredClone(action);return {status:'action',action}; }
 finish(reason:string) { if(this.ended)return;this.ended=true;this.reason=reason;const p=this.pending;this.pending=undefined;p?.reject(new Error('Episode ended: '+reason));this.stopped.abort(reason); }
}
