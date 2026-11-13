import {writeFile,access} from 'node:fs/promises';
import path from 'node:path';
import type {Robot} from '../robot/control.ts';

export async function runDiagnostic(robot:Robot,config:any) {
 const episode=robot.episode,index=episode.read().env_idx;
 const delay=(seconds:number)=>new Promise<void>((resolve,reject)=>{
  const abort=()=>{clearTimeout(timer);reject(new Error('Diagnostic episode stopped'));};
  const timer=setTimeout(()=>{episode.stopped.signal.removeEventListener('abort',abort);resolve();},seconds*1000);
  episode.stopped.signal.addEventListener('abort',abort,{once:true});
 });
 const started=Date.now();let actions=0,releasedAt:number|null=null;const actionWindows:any[]=[];
 try {
  await delay([12,0,4,8][index]);
  if(config.diagnosticReleaseFile) {
   await writeFile(path.join(episode.directory,'diagnostic_ready'),String(Date.now()));
   while(true) {
    try {await access(config.diagnosticReleaseFile);break;}catch(error:any){if(error.code!=='ENOENT')throw error;}
    await delay(1);
   }
   releasedAt=Date.now();
  }
  for(let step=0;step<3;step++) {
   const before=episode.read().step,actionStarted=Date.now();
   await robot.grip('left',step%2,1+index);
   actionWindows.push({started:actionStarted,finished:Date.now(),before,after:episode.read().step});
   actions+=1+index;
   if(step===0)await delay(index===1?10:2);
  }
  if(config.diagnosticHoldSeconds)await delay(config.diagnosticHoldSeconds);
  episode.finish('diagnostic_done');
 } finally {
  await writeFile(path.join(episode.directory,'diagnostic_result.json'),JSON.stringify({purpose:'native_infrastructure_diagnostic',index,released_at:releasedAt,action_windows:actionWindows,completed_actions:actions,last_public_step:episode.read().step,started,finished:Date.now(),reason:episode.reason}));
 }
}
