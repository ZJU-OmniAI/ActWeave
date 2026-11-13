import {randomUUID} from 'node:crypto';
import path from 'node:path';
import type {EpisodeEvidence} from '../episode/evidence.ts';
import type {JevRuntime} from '../runtime/agent.ts';

export type JevQuery = {
 questions:Record<string,{instructions:string;criteria:Record<string,string>}>;
 frames:{step:number;camera:string}[];
 action_range?:number[];
 context?:string;
 decision?:string;
 long_context?:boolean;
};

const cameras=['cam_head','cam_left_wrist','cam_right_wrist'];
const policy='Evaluate only the supplied public evidence. Caller context and candidate descriptions are hypotheses, not established facts. Compare the named steps and cameras. TCP arrival, jaw width and action names do not establish object retention, contact or task success. Choose unknown when occlusion, missing frames, conflicting views or missing alternatives prevent a supported answer. Each question is independent. Your answers are advisory; Astra owns judgement, methods, recovery and execution.';
const observationGaps:Record<string,{reason:string;next_observation:string|null}>={
 occluded:{reason:'Occlusion or overlap hides the relevant object feature or its relation to the reference.',next_observation:'Use another public view of the same state showing both the relevant feature and its reference without overlap.'},
 insufficient_detail:{reason:'The relevant features are visible but too small, blurred or unclear to distinguish the alternatives.',next_observation:'Use a clearer close view that resolves the relevant features together.'},
 missing_transition:{reason:'The supplied frames do not bracket the action or state change needed to distinguish the alternatives.',next_observation:'Compare the same public camera immediately before and after the relevant action, with the executed action interval if useful.'},
 missing_reference:{reason:'The object identity or reference needed for comparison is not visible in enough context.',next_observation:'Use a wider or complementary public view showing the target together with the distinguishing reference.'},
 not_visually_observable:{reason:'The distinction requires nonvisual evidence rather than another image.',next_observation:null},
 none:{reason:'The supplied evidence is sufficient to distinguish the alternatives.',next_observation:null},
 unknown:{reason:'No specific evidence gap can be identified.',next_observation:null}
};
const gapCriteria=Object.fromEntries(Object.entries(observationGaps).map(([key,value])=>[key,value.reason]));

function choice(value:any,criteria:Record<string,string>) {
 if(!value||value.type!=='choice'||!Object.hasOwn(criteria,value.choice)||!Number.isFinite(value.confidence)||value.confidence<0||value.confidence>1||!value.probabilities||typeof value.probabilities!=='object'||Array.isArray(value.probabilities))throw new Error('Jev returned an invalid Choice answer');
 if(Object.keys(value.probabilities).length!==Object.keys(criteria).length||Object.keys(criteria).some(key=>!Number.isFinite(value.probabilities[key])||value.probabilities[key]<0||value.probabilities[key]>1))throw new Error('Jev returned invalid option probabilities');
 return value;
}

function actionHistory(actions:any[]) {
 const fields=['left_arm_joint_state','right_arm_joint_state','left_ee_joint_state','right_ee_joint_state'];
 const columns=actions.length?fields.flatMap(key=>actions[0].action[key].map((_:number,index:number)=>`${key}[${index}]`)):[];
 const values=(row:any)=>fields.flatMap(key=>row.action[key]);
 const initial=actions.length?values(actions[0]):[];
 const runs:[number,number,[number,number][]][]=[];
 let previous=initial;
 for(const row of actions) {
  const command=values(row),updates:[number,number][]=[];
  command.forEach((value:number,index:number)=>{if(value!==previous[index])updates.push([index,value]);});
  const last=runs.at(-1);
  if(last&&last[1]===row.step&&!updates.length)last[1]=row.step+1;
  else runs.push([row.step,row.step+1,updates]);
  previous=command;
 }
 return {encoding:'command_runs_v1',instructions:'columns names the command vector. Start with initial. Each run is [start_source_step,end_source_step_exclusive,updates]. Apply its [column_index,new_value] updates to the previous vector, then use that complete vector for every source step in the run. Runs are chronological. Joint angles are radians; gripper commands are normalized openings. Values and all selected steps are retained without rounding.',columns,initial,runs};
}

export function validateQuestions(questions:JevQuery['questions']) {
 if(!questions||typeof questions!=='object'||Array.isArray(questions)||!Object.keys(questions).length)throw new Error('questions must be a nonempty named map');
 for(const [id,q] of Object.entries(questions)) {
  if(!/^[A-Za-z][A-Za-z0-9_]*$/.test(id)||id.startsWith('evidence_'))throw new Error(`questions key ${JSON.stringify(id)} must start with a letter, use letters/digits/underscores and not start with evidence_`);
  const field=`questions.${id}`;
  if(!q||typeof q!=='object'||Array.isArray(q))throw new Error(`${field} must be an object with instructions and criteria; its map key is the identifier`);
  if(typeof q.instructions!=='string'||!q.instructions.trim())throw new Error(`${field}.instructions must be a nonempty string; use instructions, not question`);
  if(!q.criteria||typeof q.criteria!=='object'||Array.isArray(q.criteria))throw new Error(`${field}.criteria must map option names to descriptions; use criteria, not choices`);
  if(!Object.hasOwn(q.criteria,'unknown'))throw new Error(`${field}.criteria.unknown is required and must describe insufficient evidence`);
  if(Object.keys(q.criteria).length<2||Object.keys(q.criteria).length>255)throw new Error(`${field}.criteria must contain 2-255 options including unknown`);
  for(const [key,value] of Object.entries(q.criteria))if(!key.trim()||typeof value!=='string'||!value.trim())throw new Error(`${field}.criteria[${JSON.stringify(key)}] requires a nonempty option name and string description`);
 }
}

export async function analyzeJev(evidence:EpisodeEvidence,config:JevRuntime,query:JevQuery,deadlineAt:number,signal:AbortSignal,toolCallId:string) {
 validateQuestions(query?.questions);
 if(!Array.isArray(query.frames)||!query.frames.length||query.frames.some(frame=>!frame||!Number.isInteger(frame.step)||frame.step< -1||!cameras.includes(frame.camera)))throw new Error('frames must name public cameras and saved steps, or -1 for current');
 if(query.context!==undefined&&typeof query.context!=='string')throw new Error('context must be a string');
 if(query.decision!==undefined&&(typeof query.decision!=='string'||!query.decision.trim()))throw new Error('decision must be a nonempty string');
 if(query.long_context!==undefined&&typeof query.long_context!=='boolean')throw new Error('long_context must be a boolean');
 const current=evidence.raw().step;
 const frames=query.frames.map(frame=>({...frame,step:frame.step===-1?current:frame.step}));
 if(frames.some(frame=>frame.step>current)||new Set(frames.map(frame=>`${frame.step}:${frame.camera}`)).size!==frames.length)throw new Error('Frames must be distinct existing observations');
 if(query.action_range&&(!Array.isArray(query.action_range)||query.action_range.length!==2||query.action_range.some(step=>!Number.isInteger(step)||step<0)||query.action_range[0]>query.action_range[1]||query.action_range[1]>current))throw new Error('action_range requires [start,end] with 0 <= start <= end <= current; source steps use [start,end)');
 const assets:Record<string,any>={},observations=new Map<number,any>();
 const images=frames.map((frame,index)=>{
  const observation=observations.get(frame.step)??evidence.raw(frame.step);observations.set(frame.step,observation);
  const asset=randomUUID();assets[asset]={type:'image',format:'png',data_base64:observation.images[frame.camera]};
  return {id:`frame_${index}`,...frame,image:asset};
 });
 const actions=query.action_range?await evidence.commands(query.action_range[0],query.action_range[1]):[];
 const evidenceCriteria:Record<string,string>={...Object.fromEntries(images.map(frame=>[frame.id,`The single frame ${frame.id} at step ${frame.step}, camera ${frame.camera}, directly supports a conclusion for this question.`])),all:'A comparison across the supplied frames is needed to support a conclusion.',unknown:'The supplied frames do not support a conclusion.'};
 const questions:Record<string,any>={};
 for(const [id,q] of Object.entries(query.questions)) {
  questions[id]={type:'choice',instructions:{policy,question:q.instructions},criteria:q.criteria};
  questions['evidence_'+id]={type:'choice',instructions:{policy,question:q.instructions,evidence_question:'Which supplied visual evidence supports a conclusion for this question? Do not assume another question has been answered.'},criteria:evidenceCriteria};
  questions['evidence__gap_'+id]={type:'choice',instructions:{policy,question:q.instructions,gap_question:'Which evidence gap, if any, most prevents answering this question for the pending decision? Assess the supplied observations independently. Choose none if the evidence is sufficient, or unknown if no specific gap can be identified.'},criteria:gapCriteria};
 }
 const history=actionHistory(actions);
 const request={model:config.model,state:{policy,instruction:evidence.raw().instruction,caller_context:query.context??'',pending_decision:query.decision??null,frames:images,public_states:[...observations.keys()].map(step=>evidence.state(step)),action_range:query.action_range??null,actions:actions.length?history:null},assets,questions};
 const remaining=deadlineAt-Date.now();
 if(remaining<=0)throw new Error('Jev call has no remaining episode time');
 const started=Date.now();
 const apiKey=process.env[config.apiKeyEnv??'OPENAI_API_KEY'];
 if(!apiKey)throw new Error('Missing Jev API credential: '+(config.apiKeyEnv??'OPENAI_API_KEY'));
 const headers:Record<string,string>={authorization:'Bearer '+apiKey,'content-type':'application/json','X-Evo-Episode-ID':path.basename(evidence.episode.directory),'X-Evo-Tool-Call-ID':toolCallId,'X-Evo-Deadline-At':String(deadlineAt/1000)};
 if(query.long_context)headers['TypeSafe-Danger-Zone-Long-Context']='true';
 const input={frames,action_range:query.action_range??null,action_steps:actions.length,action_runs:history.runs.length,long_context:query.long_context===true};
 const response=await fetch(config.endpoint,{method:'POST',headers,body:JSON.stringify(request),signal:AbortSignal.any([signal,AbortSignal.timeout(Math.max(1,Math.ceil(Math.min(config.requestSeconds*1000,remaining))))])});
 const body=await response.text();const requestId=response.headers.get('X-Router-Request-ID');
 if(!response.ok)throw new Error(JSON.stringify({tool:'jev_analyze',request_id:requestId,status:response.status,upstream_response:body,input}));
 const result=JSON.parse(body);
 if(typeof result.model!=='string'||!result.answers||Object.keys(result.answers).length!==Object.keys(questions).length||Object.keys(questions).some(key=>!Object.hasOwn(result.answers,key)))throw new Error(`Jev request ${requestId} returned incomplete answers`);
 const answers=Object.fromEntries(Object.entries(query.questions).map(([id,q])=>{
  const answer=choice(result.answers[id],q.criteria),evidence=choice(result.answers['evidence_'+id],evidenceCriteria);
  const gap=choice(result.answers['evidence__gap_'+id],gapCriteria);
  const selected=evidence.choice==='all'?frames:evidence.choice==='unknown'?[]:[frames[images.findIndex(frame=>frame.id===evidence.choice)]];
  return [id,{...answer,question:q.instructions,criteria:q.criteria,caller_criterion:q.criteria[answer.choice],
   evidence_frames:selected,evidence_selection:evidence,
   ...(answer.choice==='unknown'?{observation_gap:{...gap,...observationGaps[gap.choice]}}:{}),
   sources:{caller_criterion:'Astra-authored candidate text, not a Jev explanation',evidence_selection:'Independent Jev frame-relevance Choice, not validation of the main answer'},
   unresolved:[...(answer.choice==='unknown'?['answer_unknown']:[]),...(evidence.choice==='unknown'?['evidence_unknown']:[])]}];
 }));
 return {source:'Jev model advice; no physical state is certified',request_id:requestId,model:result.model,decision:query.decision??null,observation_step:current,answers,frames,usage:result.usage,elapsed_seconds:(Date.now()-started)/1000};
}
