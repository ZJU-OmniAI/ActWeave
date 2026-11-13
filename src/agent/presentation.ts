import type {EpisodeEvidence} from '../episode/evidence.ts';
import type {TransitionStore} from './transitions.ts';

type Snapshot=ReturnType<EpisodeEvidence['snapshot']>;
export function snapshotMetadata(snapshots:Snapshot[]) {
 return snapshots.map(item=>({...item,cameras:item.cameras.map(({png_base64,...view})=>view)}));
}
export class AstraPresentation {
 constructor(private evidence:EpisodeEvidence,private attachments:any) {}
 async pictures(snapshots:Snapshot[]) {
  const images=[];
  for(const item of snapshots)for(const view of item.cameras)images.push({step:view.step,camera:view.camera,
   attachment:await this.attachments.saveImage({data:Buffer.from(view.png_base64,'base64'),mediaType:'image/png',name:`${view.step}-${view.camera}.png`})});
  return images;
 }
 render=(value:any)=>{
  const {images=[],...payload}=value;
  if(payload.snapshots)payload.snapshots=snapshotMetadata(payload.snapshots);
  return [{type:'text',text:JSON.stringify(payload)},...images.flatMap((image:any)=>[{type:'text',text:`Original ${image.camera} at step ${image.step}`},{type:'image',attachment:image.attachment}])];
 };
 context(transitions:TransitionStore) {return {evidence:this.evidence.context(),transition:transitions.context()};}
 async show(camera:string,step=-1) {
  const view=this.evidence.image(camera,step);
  const attachment=await this.attachments.saveImage({data:Buffer.from(view.png_base64,'base64'),mediaType:'image/png',name:`${view.step}-${camera}.png`});
  return {camera,step:view.step,attachment};
 }
 async display(png:string) {
  return {attachment:await this.attachments.saveImage({data:Buffer.from(png,'base64'),mediaType:'image/png',name:'program.png'})};
 }
}
