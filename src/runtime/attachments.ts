import {randomUUID} from 'node:crypto';
import {mkdir,open,rename,rm} from 'node:fs/promises';
import path from 'node:path';
import LocalAttachments,{prepareImageFile,type PreparedImageFile} from '@deepseek-ai/dsh-attachment-local';
import type {SaveImageAttachment} from '@deepseek-ai/dsh-attachment';

export default class Attachments extends LocalAttachments {
 private active=0;
 private waiting:Array<()=>void>=[];

 private async prepare(input:SaveImageAttachment) {
  if(this.active===this.imageCompressionConcurrency)await new Promise<void>(resolve=>this.waiting.push(resolve));
  else this.active++;
  try {return await prepareImageFile(input,this.imageLimits,this.normalizationPolicy);}
  finally {
   const next=this.waiting.shift();
   if(next)next();else this.active--;
  }
 }

 private async commit(image:PreparedImageFile) {
  const target=this.imageHostPath(image.ref);
  await mkdir(path.dirname(target),{recursive:true});
  const temporary=target+'.'+randomUUID()+'.tmp';
  try {
   const handle=await open(temporary,'wx');
   try {await handle.writeFile(image.data);await handle.sync();}
   finally {await handle.close();}
   await rename(temporary,target);
  } finally {await rm(temporary,{force:true});}
  return image.ref;
 }

 async saveImage(input:SaveImageAttachment) {
  return this.commit(await this.prepare(input));
 }

 async saveImages(inputs:readonly SaveImageAttachment[]) {
  this.validateImageBatch(inputs);
  const prepared=await Promise.all(inputs.map(input=>this.prepare(input)));
  const refs=[];
  for(const image of prepared)refs.push(await this.commit(image));
  return refs;
 }
}
