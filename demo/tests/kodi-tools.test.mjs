import test from 'node:test';
import assert from 'node:assert/strict';
import { kodiToolDefinitions, executeKodiTool } from '../kodi-tools.js';

test('Kodi config ignores unrelated or malformed definitions', () => {
  assert.deepEqual(kodiToolDefinitions({kodiTools: [{name:'shutdown'}, {name:'kodi_voice_state', type:'function', parameters:{type:'object'}}]}).map(t=>t.name), ['kodi_voice_state']);
});
test('Kodi execution posts structured arguments and propagates failure', async () => {
  const output = await executeKodiTool('kodi_voice_state', {}, async (url, options) => {
    assert.equal(url, 'api/kodi');
    assert.deepEqual(JSON.parse(options.body), {name:'kodi_voice_state', arguments:{}});
    return {ok:true, json:async()=>({available:false})};
  });
  assert.equal(output, '{"available":false}');
  await assert.rejects(executeKodiTool('kodi_voice_state', {}, async()=>({ok:false,status:502,json:async()=>({detail:'offline'})})), /offline/);
});
