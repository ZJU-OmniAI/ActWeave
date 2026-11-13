import Tools from '@deepseek-ai/dsh-tools';

const directOnly=new Set(['jev_analyze','transition_observe','demonstration_read']);
const direct=new Set([...directOnly,'transition_record','transition_read','episode_snapshot','operation_experience']);

export default class ConsultationTools extends Tools {
 wireSchemas(scope) {
  const presentation=super.wireSchemas(scope);
  for(const consultation of super.schemas(scope).filter(tool=>direct.has(tool.name)))if(!presentation.knownNames.includes(consultation.name)) {
   presentation.schemas.push(consultation);
   presentation.knownNames.push(consultation.name);
  }
  return presentation;
 }
 schemas(scope) {
  return super.schemas(scope).filter(tool=>!directOnly.has(tool.name));
 }
 sdkSchemas(scope) {
  return super.sdkSchemas(scope).filter(tool=>!directOnly.has(tool.name));
 }
 collapses(name,scope,nested) {
  if(directOnly.has(name))return nested;
  return direct.has(name)?false:super.collapses(name,scope,nested);
 }
}
