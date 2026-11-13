import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import { Context } from '@deepseek-ai/cordis';
import Agent from '@deepseek-ai/dsh-agent';
import Loop from '@deepseek-ai/dsh-agent-loop';
import Llm,{createUserMessage} from '@deepseek-ai/dsh-llm';
import * as Pi from '@deepseek-ai/dsh-llm-pi-ai';
import Tools from '../tools/consultation.mjs';
import Sessions from '@deepseek-ai/dsh-session';
import Projections from '@deepseek-ai/dsh-session-projection';
import Prompt from '@deepseek-ai/dsh-system-prompt';
import Fs from '@deepseek-ai/dsh-fs-local';
import WorkingDirectory from '@deepseek-ai/dsh-working-directory';
import Attachments from '../runtime/attachments.ts';
import * as Persistence from '@deepseek-ai/dsh-session-persistence-jsonl';
import Python from '../../packages/ptc-runtime-python/index.ts';
import { mkdir,writeFile } from 'node:fs/promises';
import path from 'node:path';
import { Robot } from '../robot/control.ts';
import { Vision } from '../vision/vision.ts';
import {Demonstrations,demonstrationPrompt} from './demonstrations.ts';
import {systemPrompt} from './system-prompt.ts';
import type {AgentRuntime} from '../runtime/agent.ts';
import {TransitionStore} from './transitions.ts';
import {EpisodeEvidence} from '../episode/evidence.ts';
import {CapabilityExecution} from '../episode/execution.ts';
import {AstraPresentation} from './presentation.ts';
import {ObservationCoordinator} from './observation.ts';
import {registerRobotCapabilities} from '../capabilities/robot-tools.ts';
import {registerDecisionTools} from './decision-tools.ts';

export async function runAgent(robot:Robot,vision:Vision,root:string,config:AgentRuntime) {
 const episode=robot.episode;const workspace=path.join(episode.directory,'workspace');await mkdir(workspace,{recursive:true});
 const demonstrations=config.demonstrations?.enabled?await Demonstrations.load(root,config.demonstrations.directory!,config.taskIdentity.task):null;
 const evidence=new EpisodeEvidence(robot,vision);
 const transitions=new TransitionStore(evidence);
 const python=config.policyPython;
 const {stdout:apiReference}=await promisify(execFile)(python,[path.join(root,'packages/ptc-runtime-python/robot_api.py'),'--describe'],{timeout:10000,maxBuffer:1048576});
 const ctx=new Context();let failure:any=null,requests=0,cells=0,nudges=0;
 const episodeDeadline=config.deadlineAt;
 let recoveryNote:{text:string;recorded_at:string;recorded_step:number}|null=null;
 const budgetSnapshot=async()=>{
  const state=episode.read();
  const snapshot={sampled_at:new Date().toISOString(),wall_seconds_remaining:Math.max(0,(episodeDeadline-Date.now())/1000),cell_seconds_remaining:(ctx.ptcRuntime as Python).cellSecondsRemaining,model_requests_used:requests,model_request_limit:config.maxRequests,cells_used:cells,cell_limit:config.maxCells,native_steps_used:state.step,native_step_limit:state.step_limit};
  return {...snapshot,...await config.readUpstreamBudget()};
 };
 ctx.plugin(Fs,{cwd:workspace});
 ctx.plugin(Projections);
 ctx.plugin(Sessions);
 ctx.plugin(Llm);
 ctx.plugin(Attachments,{dshHome:episode.directory});
 ctx.plugin(Prompt,{includeHarnessIdentity:false,includeRuntimeContext:false,personaPrefix:(await systemPrompt(root,config.taskIdentity.task))+(demonstrations?'\n\n'+demonstrationPrompt:'')});
 ctx.plugin(WorkingDirectory,{defaultDirectory:workspace});
 ctx.plugin(Python,{python,robotTask:episode.read().instruction,cwd:workspace,timeoutMs:config.cellSeconds*1000,cpuSeconds:config.ptcCpuSeconds,cpuThreads:config.ptcThreads,sandboxConfig:config.sandboxConfig,launch:config.pythonLaunch});
 ctx.plugin(Tools,{mode:'ptc',maxParallelSubCalls:1});
 ctx.on('system-prompt/assemble',async (_assembly:any,_context:any,next:any)=>{
  const assembly=await next();
  for(const section of assembly.sections){
   if(section.name==='tools:sdk')section.text=apiReference;
   if(section.name==='tools:ptc-only')section.text='Use run_code for program execution and physical robot operations through the predefined robo interface inside Python; await methods marked async. Provide code and a concise description for each cell.'+(config.jev?' jev_analyze and transition_observe are direct-only read-only consultations; complete evidence enters your next model turn without running Python.':'');
  }
  for(const tool of assembly.tools)if(tool.name==='run_code'){
   tool.description='Execute a persistent Python cell for this episode. A cell may combine measurements, actions, numerical-feedback branches and recovery for one or multiple goals; cell length and observation timing are your choice. Use the documented robo methods, awaiting async operations. Use await display(image) for image output; await finish(status, summary) accepts only failure or stuck to stop the policy. Native-confirmed success ends the episode automatically. Variables and helpers persist. Await every operation; do not start background tasks. Print selected feedback or leave a trailing expression. Camera and display requests are delivered after the cell. The code runs in the episode workspace. '+(config.pythonLaunch?.description??'Filesystem and network access are restricted by the configured sandbox. Use only the documented robot interface; system, shell, network and simulator access are prohibited.');
   tool.parameters.properties.code.description='Python cell source using the predefined robo, np, math, time, log, display and finish bindings. Top-level await is supported; top-level return is invalid.';
   tool.parameters.properties.description.description='A concise description of this Python program.';
  }
  return assembly;
 });
 ctx.plugin(Pi,{providers:{evo:{api:'openai-completions',apiKeyEnv:'OPENAI_API_KEY',baseURL:process.env.OPENAI_BASE_URL,maxRequestImageBytes:Number.MAX_SAFE_INTEGER,requestImagePixelBudget:Number.MAX_SAFE_INTEGER,requestImageMaxBytes:Number.MAX_SAFE_INTEGER,
  timeoutMs:(config.apiRequestSeconds+30)*1000,streamIdleTimeoutMs:(config.apiRequestSeconds+30)*1000,retryPolicy:{mode:'normal',maxRetries:0},models:[{id:config.model,name:config.model,contextWindow:262144,maxTokens:config.maxTokens,input:['text','image'],reasoningEfforts:{high:'high'}}]}}});
 ctx.plugin(Agent);
 ctx.plugin(Persistence,{root:path.join(episode.directory,'sessions'),compression:'none'});
 ctx.plugin(Loop,{agents:[],maxParallelToolCalls:1});
 ctx.on('llm/stream',async function* (_options:any,next:any) {
  const calls=new Map<number,{raw:string;ended:boolean}>();
  for await(const chunk of next()) {
   if(chunk.type==='block-start'&&chunk.blockType==='tool-call')calls.set(chunk.index,{raw:'',ended:false});
   if(chunk.type==='tool-call-delta')calls.get(chunk.index)!.raw+=chunk.argumentsDelta;
   if(chunk.type==='block-end'&&chunk.block.type==='tool-call')calls.get(chunk.index)!.ended=true;
   if(chunk.type==='finish'&&chunk.reason.kind==='max-tokens'&&calls.size) {
    const complete=[...calls.values()].every(call=>{
     if(!call.ended)return false;
     try {const args=JSON.parse(call.raw);return args!==null&&typeof args==='object'&&!Array.isArray(args);}
     catch {return false;}
    });
    if(complete){yield {...chunk,reason:{kind:'tool-calls'}};continue;}
   }
   yield chunk;
  }
 });
 ctx.on('session/event',(_session:any,event:any)=>evidence.sessionEvent(event));
 ctx.on('agent/error',(event:any)=>{failure=String(event.error??event);episode.finish('agent_error');});
 ctx.on('agent/pre-step',async (_event:any,next:any)=>{
  if(episode.ended) return {kind:'reject'};
  if(config.maxRequests!==null&&requests>=config.maxRequests){episode.finish('request_budget');return {kind:'reject'};}
  requests++;return next();
 });
 ctx.on('tools/pre-execute',async (exec:any,next:any)=>{
  if(exec.name==='run_code'&&++cells>config.maxCells){episode.finish('cell_budget');return {kind:'deny',reason:'cell_budget'};}
  return next();
 });
 await new Promise<void>((resolve,reject)=>{const timer=setTimeout(()=>reject(new Error('dsh services did not mount')),10000);ctx.inject(['agentLoop','tools','attachments','workingDirectory','ptcRuntime'],()=>{clearTimeout(timer);resolve();});});
 const presentation=new AstraPresentation(evidence,ctx.attachments);
 const execution=new CapabilityExecution(ctx.tools,evidence,()=>transitions.association());
 const observer=new ObservationCoordinator(evidence,transitions,presentation,config);
 const tool=execution.register;
 if(demonstrations)tool('demonstration_read','Read official training-video references, never live observations. Omit id for the supplied reference catalog. With id, frames are source video-frame indices, NOT native physics steps. Default selects the overview. Select up to 12 frames and any available cameras to examine a contact or method. Returned RGB and times describe another episode, not current geometry or verified reasoning.',{
  id:{type:'string'},frames:{type:'array',items:{type:'integer'}},cameras:{type:'array',items:{type:'string',enum:['cam_head','cam_left_wrist','cam_right_wrist']}}
 },(a:any)=>demonstrations.read(a,ctx.attachments),(value:any)=>value);
 registerRobotCapabilities(tool,evidence,presentation,budgetSnapshot);
 registerDecisionTools(tool,evidence,transitions,presentation,observer,Boolean(config.jev),root);
 tool('recovery_note','Optionally replace your current episode recovery note, or read it without text. It is your revisable account, not verified physical state. Recording time and step describe the update. The host retains it across Python restarts; it does not select actions or decide when to stop.',{text:{type:'string'}},(a:any)=>{
  if(a.text!==undefined)recoveryNote={text:a.text,recorded_at:new Date().toISOString(),recorded_step:episode.read().step};
  return recoveryNote;
 });
 const handle=await ctx.agents.create({sessionId:path.basename(episode.directory) as any,meta:{cwd:workspace},agentOptions:{provider:'evo',model:config.model,reasoningEffort:config.reasoningEffort,maxTokens:config.maxTokens}});
 const stop=()=>handle.agent.cancel({kind:'hook',reason:episode.reason});
 episode.stopped.signal.addEventListener('abort',stop,{once:true});
 const timer=setTimeout(()=>episode.finish('episode_deadline'),Math.max(0,episodeDeadline-Date.now()));
 ctx.on('system-prompt/assemble',async(_assembly:any,_context:any,next:any)=>{
  const assembly=await next();
  assembly.contexts.push({name:'episode-feedback',text:JSON.stringify({budget:await budgetSnapshot(),recovery_note:{source:'Your own revisable note; not verified physical state',current:recoveryNote}}),interpolate:false});
  assembly.contexts.push({name:'decision-evidence',text:JSON.stringify(presentation.context(transitions)),interpolate:false});
  return assembly;
 });
 try {
  if(episode.ended)return;
  const initialStep=evidence.step;
  const initialImages=await presentation.pictures([evidence.snapshot(initialStep,Object.keys(evidence.raw(initialStep).images))]);
  handle.agent.followup(createUserMessage({source:{kind:'user'},content:[...(demonstrations?await demonstrations.initial(ctx.attachments):[]),{type:'text',text:[demonstrations?'# Current live episode':'# Current episode',`Robot: dual-arm ARX-X5; RGB only; no VLA.`,config.taskIdentity?`Task identity: ${JSON.stringify(config.taskIdentity)}`:'',`Native instruction (original wording): ${JSON.stringify(episode.read().instruction)}`,`Initial public state: ${JSON.stringify(robot.publicState())}`,`Jev visual consultation: ${config.jev?'enabled ('+config.jev.model+'); jev_analyze and transition_observe are directly callable only, outside Python':'disabled for this profile'}.`,`Current IK planning ranges (radians, an empirical search domain, not simulator hard limits): ${JSON.stringify(robot.kin.chain.map(j=>j.limits))}`,`Limits: ${config.maxRequests===null?'unlimited':config.maxRequests} model requests, ${config.maxCells} Python cells (including observation-only cells), ${config.episodeSeconds} seconds wall time including thinking, ${config.cellSeconds} seconds per cell, ${episode.read().step_limit} native actions. Limits are not reset by a Python restart.`,`Attached cameras show the initial state of this episode, at original 640x480 pixels. They are the same arrays returned by await robo.image(camera,step=0). Row/col coordinates refer directly to these pixels.`].filter(Boolean).join('\n')},...initialImages.flatMap(({camera,attachment})=>[{type:'text' as const,text:camera},{type:'image' as const,attachment}])]}));
  await handle.agent.whenIdle();
  while(!episode.ended&&!failure&&nudges<config.maxNudges){
   nudges++;
   handle.agent.followup(createUserMessage({source:{kind:'user'},content:[{type:'text',text:'The policy remains active. Native success has not ended the episode; observation/computation alone does not trigger a new native check. You choose actions, observations, program composition and recovery. run_code executes Python; finish(status, summary) with status failure or stuck records your decision to stop without a physical action. An idle text reply does not finish the episode.'}]}));
   await handle.agent.whenIdle();
  }
  if(!episode.ended)episode.finish(failure?'agent_error':'policy_return');
 } finally {
  clearTimeout(timer);episode.stopped.signal.removeEventListener('abort',stop);await handle.dispose();await ctx.fiber.dispose();
  await writeFile(path.join(episode.directory,'agent_result.json'),JSON.stringify({reason:episode.reason,failure,requests,cells,nudges,harness_version:'4',...(demonstrations?{demonstrations:demonstrations.catalog()}:{}),evidence:evidence.export(),transitions:transitions.export()}));
 }
}
