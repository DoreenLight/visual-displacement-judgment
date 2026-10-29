// Read-only camera/UI integration check. No training or recording request is sent.
const {chromium}=require(process.env.FEEDBACK_PLAYWRIGHT_PATH||'playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage({viewport:{width:1440,height:1100}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto('http://localhost:8090/',{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>document.querySelectorAll('.task-card').length===8);
 await page.waitForFunction(()=>document.querySelector('.video img').naturalWidth>0);
 const first=await page.evaluate(()=>fetch('/telemetry').then(r=>r.json()));
 await page.waitForTimeout(350);
 const second=await page.evaluate(()=>fetch('/telemetry').then(r=>r.json()));
 if(second.frame_id<=first.frame_id)throw Error('Camera analysis did not advance');
 if(second.capture_age_ms>500||second.result_age_ms>1000)throw Error('Stale camera results');
 if(errors.length)throw Error(errors.join('\n'));
 await page.screenshot({path:path.join(__dirname,'eight_task_interface.png'),fullPage:true,mask:[page.locator('.video img')],maskColor:'#10202b'});
 const report={checked_at:new Date().toISOString(),camera:second.camera,selected_task:second.task_id,
   eight_options:await page.locator('#taskSelect option').count(),eight_cards:await page.locator('.task-card').count(),
   advancing_frame_ids:[first.frame_id,second.frame_id],fps:second.fps,inference_ms:second.inference_ms,
   host_receive_to_output_ms:second.processing_latency_ms,capture_age_ms:second.capture_age_ms,result_age_ms:second.result_age_ms,
   page_errors:errors,recording:second.recording,software_tests:'11 protocol/export + 9 regression; 45 mocked UI operations',
   full_human_protocol_collected:false,formal_accuracy_validation:'BLOCKED: no independent reference',
   screenshot_note:'Camera pixels masked; interface only. Actual raw RGB-D is saved only on user-started trials.'};
 fs.writeFileSync(path.join(__dirname,'eight_task_engineering_verification.json'),JSON.stringify(report,null,2));
 console.log(JSON.stringify(report));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
