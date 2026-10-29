// Browser regression test. All endpoints are mocked: no camera or recording is touched.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.FEEDBACK_PLAYWRIGHT_PATH || 'playwright');

(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  try {
    const page = await browser.newPage();
    const errors = [];
    const commands = [];
    const telemetry = {status:'ready', calibrated:true, training:false, recording:false,
      result_age_ms:10, headline:'test ready', detail:'mock observation', bend_deg:0,
      target_deg:20, shoulder_mm:0, elbow_deg:0, shoulder_alert:false, elbow_alert:false};
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/*', async route => {
      const request = route.request();
      const pathname = new URL(request.url()).pathname;
      if (pathname === '/') {
        return route.fulfill({contentType:'text/html; charset=utf-8',
          body:fs.readFileSync(path.join(__dirname,'live_feedback_v2.html'),'utf8')});
      }
      if (pathname === '/video_feed') return route.abort();
      if (pathname === '/telemetry') {
        return route.fulfill({contentType:'application/json',body:JSON.stringify(telemetry)});
      }
      if (request.method() === 'POST') {
        const action = pathname.slice(1);
        const payload = request.postDataJSON();
        commands.push(action);
        if (action === 'train') telemetry.training = true;
        if (action === 'pause') telemetry.training = false;
        if (action === 'record') telemetry.recording = true;
        if (action === 'stop_record') telemetry.recording = false;
        if (action === 'target') telemetry.target_deg = payload.value;
        if (action === 'calibrate') telemetry.calibrated = false;
        return route.fulfill({contentType:'application/json',body:JSON.stringify({accepted:action})});
      }
      return route.fulfill({status:404});
    });
    await page.goto('http://localhost:8090/',{waitUntil:'domcontentloaded'});
    await page.waitForFunction(() => !document.querySelector('#train').disabled);
    await page.locator('#train').click();
    await page.waitForFunction(() => document.querySelector('#train').textContent === '训练进行中');
    await page.getByRole('button',{name:'暂停训练',exact:true}).click();
    await page.waitForFunction(() => !document.querySelector('#train').disabled);
    await page.locator('#record').click();
    await page.waitForFunction(() => document.querySelector('#record').textContent === '停止录制');
    await page.locator('#record').click();
    await page.waitForFunction(() => document.querySelector('#record').textContent === '开始录制');
    await page.locator('#target').fill('25');
    await page.getByRole('button',{name:'设置',exact:true}).click();
    await page.getByRole('button',{name:'重新标定',exact:true}).click();
    await page.waitForFunction(() => document.querySelector('#train').disabled);
    await page.getByRole('button',{name:'结束并关闭摄像头',exact:true}).click();
    await page.waitForFunction(() => document.querySelector('#status').textContent === '已结束');
    assert.deepEqual(commands,['train','pause','record','stop_record','target','calibrate','stop']);
    assert.equal(telemetry.target_deg,25);
    assert.deepEqual(errors,[]);
    console.log('PASS: all seven command paths; no browser script errors.');
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error);process.exitCode=1;});
