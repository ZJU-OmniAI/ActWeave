import {readFile,stat} from 'node:fs/promises';
import path from 'node:path';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';

const execute=promisify(execFile);
type Video={file:string;frames:number;fps:number;width:number;height:number};
type Reference={id:string;tasks:string[];source_task:string;source:{repository:string;revision:string;episode_index:number};overview_frames:number[];videos:Record<string,Video>;method:string;applicability:string;selection_reason:string};
export const demonstrationPrompt=`# Official training references

The initial reference messages contain fixed, curated official training episodes packaged with this harness, not actions executed in this live episode. Their frames are time-ordered video observations; no expert reasoning, metric state, action targets, calibration or hidden object poses are supplied. Overview frames were selected to expose useful stages. The method and applicability notes are curator interpretations, not expert reasoning or certified contact events. Use demonstration_read to inspect intermediate frames or a wrist view when an important transition is missing from the overview. Only this task's fixed references are available; there is no external training database or runtime retrieval.

Use a reference as a candidate method and an example of visible object change. Identify the useful contact region, support arrangement, hand roles and subgoal transitions, then ground the objects, coordinates and required relations in the current live scene. The reference's layout and camera geometry may differ. Do not copy its timing, coordinates, object count or order over the live instruction. Decide which conditions make the method applicable and change it when live feedback contradicts those conditions.

reference_id/video_frame/video_seconds refer only to the training video. live step/camera and tool receipts refer only to the current episode. Never pass reference frame indices to live observation or Jev tools. Reference success does not establish live retention, support, insertion or completion. Ask Jev about current before/after evidence. An occluded relation remains unresolved even when the reference eventually succeeds. Preserve Astra's choice to adapt, replace or reject the demonstrated method; use the strategy and feedback loop above for states absent from the examples.`;

export class Demonstrations {
 private constructor(private directory:string,private references:Reference[],private task:string) {}
 static async load(root:string,directory:string,task:string) {
  const manifest=JSON.parse(await readFile(path.join(root,'configs/demonstrations/official-training.json'),'utf8'));
  if(manifest.schema_version!==2||manifest.repository!=='RoboDojo-Benchmark/RoboDojo'||!manifest.revision||!Array.isArray(manifest.tasks)||!Array.isArray(manifest.without_demonstrations))throw new Error('Invalid curated training reference catalog');
  const matches=manifest.tasks.filter((row:any)=>row.tasks.includes(task));
  if(manifest.without_demonstrations.includes(task)){
   if(matches.length)throw new Error('A no-demonstration task has training references: '+task);
   return null;
  }
  if(matches.length!==1)throw new Error('No unique curated demonstration setting for '+task);
  const setting=matches[0];
  if(task!==setting.source_task&&task!==setting.source_task+'_random')throw new Error('Cross-task demonstrations are not supported');
  const records:Reference[]=setting.episodes.map((episode:any)=>{
   const id=setting.source_task+'-'+String(episode.episode_index).padStart(7,'0');
   return {id,tasks:setting.tasks,source_task:setting.source_task,
    source:{repository:manifest.repository,revision:manifest.revision,episode_index:episode.episode_index},
    overview_frames:episode.overview_frames,method:episode.method,applicability:episode.applicability,selection_reason:episode.selection_reason,
    videos:Object.fromEntries(manifest.cameras.map((camera:string)=>[camera,{file:id+'/'+camera+'.mp4',frames:episode.frames,fps:episode.fps,width:640,height:480}]))};
  });
  if(!records.length||new Set(records.map(row=>row.id)).size!==records.length)throw new Error('Distinct curated training references are required for '+task);
  for(const row of records){
   if(typeof row.id!=='string'||!/^[-A-Za-z0-9_]+$/.test(row.id)||!row.source||row.source.repository!=='RoboDojo-Benchmark/RoboDojo'||!row.source.revision||!Array.isArray(row.overview_frames)||!row.overview_frames.length)throw new Error('Reference source or overview is missing');
   if(!row.videos?.cam_head)throw new Error('A reference requires the head camera');
   for(const [camera,video] of Object.entries(row.videos)){
    if(!['cam_head','cam_left_wrist','cam_right_wrist'].includes(camera)||!Number.isInteger(video.frames)||video.frames<1||!Number.isFinite(video.fps)||video.fps<=0||path.isAbsolute(video.file)||video.file.split(/[\\/]/).includes('..'))throw new Error('Invalid reference video metadata');
    if(!(await stat(path.join(directory,video.file))).isFile())throw new Error('Missing packaged reference video: '+video.file);
   }
   if(row.overview_frames.some(frame=>!Number.isInteger(frame)||frame<0||frame>=row.videos.cam_head.frames))throw new Error('Invalid reference overview frame');
  }
  return new Demonstrations(directory,records,task);
 }
 catalog() {
  return this.references.map(({videos,...reference})=>({...reference,cameras:Object.fromEntries(Object.entries(videos).map(([camera,{file,...metadata}])=>[camera,metadata])),
   target_task:this.task,relationship:this.task===reference.source_task?'same_task':'standard_variant_reference',kind:'reference_not_live',sampling:'Curator-selected stage overview; method notes are interpretations, not ground truth',available_signals:['RGB','video_frame','video_seconds','curator_method_notes'],unavailable_signals:['robot_state','action_targets','camera_calibration','object_ground_truth','expert_reasoning']}));
 }
 async read(query:{id?:string;frames?:number[];cameras?:string[]},attachments:any) {
  if(query.id===undefined){
   if(query.frames!==undefined||query.cameras!==undefined)throw new Error('Select a reference id before selecting frames or cameras');
   return [{type:'text',text:JSON.stringify({references:this.catalog()})}];
  }
  const reference=this.references.find(row=>row.id===query.id);
  if(!reference)throw new Error('Unknown supplied training reference: '+query.id);
  const frames=query.frames??reference.overview_frames,cameras=query.cameras??['cam_head'];
  if(!Array.isArray(frames)||!frames.length||frames.length>12||new Set(frames).size!==frames.length||frames.some(frame=>!Number.isInteger(frame)||frame<0))throw new Error('Select 1-12 distinct nonnegative video-frame indices');
  if(!Array.isArray(cameras)||!cameras.length||new Set(cameras).size!==cameras.length||cameras.some(camera=>!reference.videos[camera]))throw new Error('Select distinct available reference cameras');
  const content:any[]=[{type:'text',text:JSON.stringify({reference:this.catalog().find(row=>row.id===reference.id),selected_video_frames:frames,source:'Historical official training video, not current state or expert reasoning'})}];
  for(const frame of frames)for(const camera of cameras){
   const video=reference.videos[camera];
   if(frame>=video.frames)throw new Error('Reference frame exceeds '+camera+' video length');
   const {stdout}=await execute('ffmpeg',['-v','error','-threads','1','-filter_threads','1','-i',path.join(this.directory,video.file),'-vf',`select=eq(n\\,${frame})`,'-frames:v','1','-f','image2pipe','-c:v','png','pipe:1'],{timeout:20000,maxBuffer:8*1024*1024,encoding:'buffer'});
   if(!stdout.length)throw new Error('Reference video produced no image: '+reference.id+'/'+camera+'/'+frame);
   const attachment=await attachments.saveImage({data:stdout,mediaType:'image/png',name:`reference-${reference.id}-${frame}-${camera}.png`});
   content.push({type:'text',text:JSON.stringify({kind:'reference_not_live',reference_id:reference.id,video_frame:frame,video_seconds:frame/video.fps,camera})},{type:'image',attachment});
  }
  return content;
 }
 async initial(attachments:any) {
  const content:any[]=[{type:'text',text:'# Historical training references\nStudy the visible method and effects, then adapt them to the live episode that follows. These images are not current observations.'}];
  for(const reference of this.references)content.push(...await this.read({id:reference.id},attachments));
  return content;
 }
}
