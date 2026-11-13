import {createServer} from 'node:http';
import {Agent,setGlobalDispatcher} from 'undici';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {randomUUID} from 'node:crypto';
import {Episode} from './bridge/episode.ts';
import {Kinematics} from './robot/kinematics.ts';
import {Robot} from './robot/control.ts';
import {Vision} from './vision/vision.ts';
import {agentRuntime} from './bridge/agent-runtime.ts';
import {runAgent} from './agent/host.ts';

const root=fileURLToPath(new URL('../',import.meta.url));
if(!process.env.EVO_CONFIG)throw new Error('EVO_CONFIG must name a runtime JSON file');
const config=JSON.parse(await readFile(path.resolve(root,process.env.EVO_CONFIG),'utf8'));
for(const key of ['policyPython','robotCalibration','sandboxConfig','outputRoot'])config[key]=path.resolve(root,config[key]);
if(config.demonstrations?.enabled)config.demonstrations.directory=path.resolve(root,config.demonstrations.directory);
if(config.maxRequests!==null)throw new Error('Cumulative model request budgets must be unlimited: maxRequests=null');
if(!process.env.OPENAI_BASE_URL||!process.env.OPENAI_API_KEY)throw new Error('OPENAI_BASE_URL and OPENAI_API_KEY are required');
if(config.jev&&!process.env.JEV_API_KEY)throw new Error('JEV_API_KEY is required when Jev is enabled');
const apiWaitMs=(config.apiRequestSeconds+30)*1000;
setGlobalDispatcher(new Agent({headersTimeout:apiWaitMs,bodyTimeout:apiWaitMs}));
const kin=new Kinematics(config.robotCalibration);
let task:string|undefined;
let entry:{episode:Episode,agent:Promise<void>,nativeEnded:boolean}|undefined;
async function waitForAgent() {
 let timer:ReturnType<typeof setTimeout>|undefined;
 try {await Promise.race([entry?.agent,new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('Agent cleanup exceeded 10 seconds')),10000);})]);}
 finally {clearTimeout(timer);}
}
async function observe(data:any) {
 if(data.env_idx!==0)throw new Error('The policy bridge accepts one environment');
 if(!task)throw new Error('Reset the policy before sending observations');
 if(entry?.nativeEnded)throw new Error('Reset the policy before starting another episode');
 if(!entry){
  const episode=new Episode(path.join(config.outputRoot,'episodes',randomUUID()));
  await mkdir(episode.directory,{recursive:true});
  await episode.update(data);
  const robot=new Robot(episode,kin);
  const runtime=agentRuntime(config,task,Date.now()+config.episodeSeconds*1000);
  const agent=runAgent(robot,new Vision(robot,`http://127.0.0.1:${config.ports.perception}`),root,runtime).catch(async error=>{
   episode.finish('host_error');await writeFile(path.join(episode.directory,'host_error.txt'),String(error.stack??error));
  });
  entry={episode,agent,nativeEnded:false};
 } else await entry.episode.update(data);
}
const server=createServer(async(req,res)=>{
 try {
  let body='';for await(const chunk of req)body+=chunk;
  const data=body?JSON.parse(body):{};let result:any={ok:true};
  if(req.url==='/reset') {
   if(typeof data.task!=='string'||!data.task.trim())throw new Error('Reset requires a task name');
   if(entry&&!entry.nativeEnded)throw new Error('The previous episode has not ended');
   await waitForAgent();entry=undefined;task=data.task;
  } else if(req.url==='/observation')await observe(data);
  else if(req.url==='/action')result=entry!.episode.take();
  else if(req.url==='/end'){
   const episode=entry!.episode;episode.finish('native_episode_end');
   if(data.final_observation){await episode.update(data.final_observation);delete data.final_observation;}
   await writeFile(path.join(episode.directory,'native_result.json'),JSON.stringify(data));entry!.nativeEnded=true;
   await waitForAgent();
  } else if(req.url!=='/health')throw new Error('Unknown route');
  res.writeHead(200,{'content-type':'application/json'});res.end(JSON.stringify(result));
 }catch(error){res.writeHead(500,{'content-type':'application/json'});res.end(JSON.stringify({error:String(error)}));}
});
server.listen(config.ports.bridge,'127.0.0.1');
for(const signal of ['SIGTERM','SIGINT'] as const)process.once(signal,()=>{
 entry?.episode.finish(signal);server.close();server.closeAllConnections();
 void waitForAgent().then(()=>process.exit(0),error=>{process.stderr.write(String(error)+'\n');process.exit(1);});
});
