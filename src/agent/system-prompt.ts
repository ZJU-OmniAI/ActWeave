import {readFile} from 'node:fs/promises';
import path from 'node:path';

type TaskPrompt = {tasks:string[];text:string};

export async function systemPrompt(root:string,task:string) {
 const [shared,source]=await Promise.all([
  readFile(path.join(root,'src/agent/prompt.txt'),'utf8'),
  readFile(path.join(root,'configs/agents/task-prompts.json'),'utf8'),
 ]);
 const entries:TaskPrompt[]=JSON.parse(source);
 const guidance=entries.filter(entry=>entry.tasks.includes(task)).map(entry=>entry.text);
 return [shared.trimEnd(),...(guidance.length?['# Static task requirements\n\n'+guidance.join('\n\n')]:[])].join('\n\n');
}
