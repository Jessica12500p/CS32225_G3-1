export const $=id=>document.getElementById(id);
export const escapeHTML=value=>String(value).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
export async function api(path,options){const response=await fetch(path,options);if(!response.ok){let message;try{message=(await response.json()).error;}catch{message=response.statusText;}throw new Error(message||'请求失败');}return response.json();}
export function showError(error){$('error').textContent=`读取失败：${error.message}`;$('error').hidden=false;}
export function clearError(){$('error').hidden=true;}
export function safe(fn){return async(...args)=>{try{clearError();await fn(...args);}catch(error){showError(error);}};}
