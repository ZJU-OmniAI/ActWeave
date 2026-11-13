import {readFile} from 'node:fs/promises';
import path from 'node:path';
import type {Register} from '../episode/execution.ts';
import {cameraNames,publicCameras,type EpisodeEvidence} from '../episode/evidence.ts';
import type {TransitionStore} from './transitions.ts';
import type {AstraPresentation} from './presentation.ts';
import type {ObservationCoordinator} from './observation.ts';

export function registerDecisionTools(tool:Register,evidence:EpisodeEvidence,transitions:TransitionStore,presentation:AstraPresentation,observer:ObservationCoordinator,jevEnabled:boolean,root:string) {
 const id={type:'string',required:true};
 const camera={type:'string',enum:publicCameras,required:true};
 const cameras={type:'array',items:{type:'string',enum:publicCameras}};
 const frames={type:'array',items:{type:'object',additionalProperties:false,properties:{step:{type:'integer',required:true},camera}}};
 if(jevEnabled)tool('jev_analyze','Read-only Jev consultation on selected public frames, states and executed actions. Call this tool directly, outside run_code; it is unavailable inside Python. Complete answers enter Astra context before its next decision. decision names the pending choice; questions identify individual observable facts with distinct criteria including unknown. Caller context and criteria are hypotheses. Returns native Choice distributions, caller-authored criterion text, and an independent frame-relevance Choice. Unknown answers also include an independent observation_gap Choice and, when applicable, a suggested kind of observation. This is advice, not verified state. No physics, automatic observation, recovery or action gate. long_context=true explicitly requests 262k context (slower, lower accuracy); default false. Errors preserve original status/body and input scope.',{
  questions:{type:'object',required:true,additionalProperties:true,description:'Nonempty map: {"object_state":{"instructions":"Is the named object retained at the selected step?","criteria":{"retained":"The object is visibly between the fingers.","free":"The object is visibly separate.","unknown":"Evidence does not resolve retention."}}}. The map key is the identifier; each value needs instructions and criteria, not question/choices. Use 2-255 nonempty option descriptions including unknown; IDs start with a letter and use letters/digits/underscores, excluding the evidence_ prefix.'},
  frames:{type:'array',required:true,items:{type:'object',additionalProperties:false,properties:{step:{type:'integer',required:true},camera}}},
  action_range:{type:'array',items:{type:'integer'}},context:{type:'string'},decision:{type:'string'},long_context:{type:'boolean'}
 },(a:any,e:any)=>observer.analyze(a,e));
 tool('episode_snapshot','Read one immutable public step: original camera images, calibration and robot state. Direct calls display images by default. display=false returns PNG data without model images, for Python computation. Does not advance physics.',{step:{type:'integer'},cameras,display:{type:'boolean'}},async(a:any)=>{
  const snapshots=[evidence.snapshot(a.step??-1,cameraNames(a.cameras??publicCameras))];
  return {snapshots,images:a.display===false?[]:await presentation.pictures(snapshots),source:'Public observations and calibration at the same recorded step; no object truth'};
 },presentation.render);
 tool('transition_record','Record the working hypothesis for a cross-turn operation, or replace it when your method or phase changes. Each description update starts a new revision and phase at the current step. objects map instance names to descriptions; basis, goal_delta, expected_delta, preserve[], decision and anchors[{step,camera}] are required. Optional basis_receipts references existing measurement/execution receipts. Optional target_anchors defaults to the first anchors and persists unless explicitly replaced; [] clears it. Optional depends_on=[{id,revision?}] pins an existing operation revision and includes its historical evidence. Optional watch={questions,cameras?,long_context}; questions use Jev instructions/criteria with unknown. active=true associates subsequent physical calls; active=false removes this association only if this id is active. No stage inference, automatic consultation, action or gate.',{
  id,description:{type:'object',additionalProperties:true},active:{type:'boolean'}
 },(a:any)=>transitions.record(a.id,a.description,a.active));
 tool('transition_read','Read original transition revisions, physical call receipts and observation results, retained across Python restarts. Omit id to list records and read episode-level evidence, including calls made without active transitions. In that mode after_observation selects consultation ids. Default revision is current; older revisions and optional after_receipt/after_observation cursors are explicit. This is recorded evidence, not current physical truth.',{
  id:{type:'string'},revision:{type:'integer'},after_receipt:{type:'integer'},after_observation:{type:'integer'}
 },(a:any)=>transitions.read(a.id,a.revision,a.after_receipt,a.after_observation));
 tool('transition_observe','Direct-only read-only observation at your chosen boundary after a Python cell. Reuse the named watch and automatically package operation-boundary, gripper/failure/direction-change and representative intermediate frames, fixed references, unresolved history and declared dependencies. Every historical reference needed by Jev is supplied inline. Return original images/receipts and explicit coverage with the complete answer. Optional frames ADD chosen intermediate views; they do not remove anchors. No token/image budget precheck, automatic polling, phase transition, action or recovery. Repeating identical evidence reuses its successful answer, explicitly labelled reused; errors are exposed without retries. Use jev_analyze for arbitrary one-off questions or a fresh repeated consultation. long_context is explicit.',{
  id,frames,long_context:{type:'boolean'}
 },(a:any,e:any)=>observer.observe(a.id,a.frames,a.long_context,e),presentation.render);
 tool('operation_experience','Read mechanism guidance or historical experience. Mechanism entries are authored decision guidance; historical entries cite user-approved v2 168, v3 168 and Jev 84 evidence. Omit id for the catalog. Advice is not a mandatory recipe or an action gate; select or reject it using the current evidence.',{id:{type:'string'}},async(a:any)=>{
  const entries=JSON.parse(await readFile(path.join(root,'configs/agents/operation-experience.json'),'utf8'));
  if(a.id===undefined)return entries.map((entry:any)=>({id:entry.id,title:entry.title,applies:entry.applies,source_type:entry.source_type}));
  const entry=entries.find((item:any)=>item.id===a.id);
  if(!entry)throw new Error('Unknown experience id: '+a.id);
  return entry;
 });
}
