// All collection actions are intercepted. This test never records a person or opens the camera.
const {chromium}=require(process.env.FEEDBACK_PLAYWRIGHT_PATH||'playwright');
const fs=require('fs'),path=require('path');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage({viewport:{width:1440,height:1100}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));const commands=[];
 const catalog=[['S1','自然坐姿静止'],['S2','握柄姿态静止'],['FE','标准腕屈伸'],['RU','标准桡偏／尺偏'],['CIRCLE','腕部复合回转'],['SHOULDER','肩部诱导代偿'],['ELBOW','肘部诱导代偿'],['COMBINED','肩肘协同代偿']].map((x,i)=>({id:i+1,code:x[0],name:x[1],group:'测试',instruction:'软件测试场景',duration_s:i<2?30:null,target_reps:i>=2&&i<=4?10:0}));
 const state={task_id:1,task_catalog:catalog,participant_id:'P001',status:'ready',headline:'软件测试',detail:'fixture',
   result_age_ms:0,capture_age_ms:0,calibrated:true,training:false,recording:false,finalizing:false,paused:false,
   elapsed_s:0,manual_count:0,manual_clockwise:0,manual_counterclockwise:0,direction:'clockwise',completed_runs:[]};
 await page.route('http://127.0.0.1:8090/**',async route=>{
   const req=route.request(),url=new URL(req.url());
   if(url.pathname==='/')return route.fulfill({status:200,contentType:'text/html; charset=utf-8',body:fs.readFileSync(path.join(__dirname,'live_feedback_v3.html'),'utf8')});
   if(url.pathname==='/telemetry')return route.fulfill({json:state});
   if(url.pathname==='/video_feed')return route.fulfill({status:200,body:''});
   if(req.method()==='POST'){
     const action=url.pathname.slice(1),data=req.postDataJSON();commands.push({action,data});
     if(action==='select_task'){state.task_id=data.task_id;state.participant_id=data.participant_id;state.manual_count=0;state.direction='clockwise';}
     if(action==='train'){state.training=true;state.recording=true;}
     if(action==='pause')state.paused=true;if(action==='resume')state.paused=false;
     if(action==='finish'){state.training=false;state.recording=false;state.paused=false;}
     if(action==='mark_rep')state.manual_count++;
     if(action==='switch_direction')state.direction='counterclockwise';
     return route.fulfill({json:{accepted:action}});
   }
   return route.fulfill({status:404});
 });
 await page.goto('http://127.0.0.1:8090/');
 for(let id=1;id<=8;id++){
   await page.selectOption('#taskSelect',String(id));await page.waitForFunction(id=>document.getElementById('taskSelect').value===String(id)&&!document.getElementById('start').disabled,id);
   await page.click('#start');await page.waitForFunction(()=>!document.getElementById('pause').disabled);
   await page.click('#pause');await page.getByRole('button',{name:'继续',exact:true}).waitFor();await page.click('#pause');
   await page.getByRole('button',{name:'暂停',exact:true}).waitFor();
   if([3,4,5].includes(id)){await page.click('#markRep');await page.waitForFunction(()=>document.getElementById('counts').textContent.includes('人工'));
     if(id===5){await page.click('#direction');await page.waitForFunction(()=>document.getElementById('direction').disabled);}}
   await page.click('#finish');await page.waitForFunction(()=>!document.getElementById('start').disabled);
 }
 await page.click('#calibrate');await page.waitForTimeout(350);
 state.result_age_ms=1500;state.estimate_valid=false;await page.waitForTimeout(350);
 if(!await page.isDisabled('#start'))throw Error('Stale result allowed start');
 if(errors.length)throw Error(errors.join('\n'));
 if(commands.filter(x=>x.action==='train').length!==8||commands.filter(x=>x.action==='finish').length!==8)throw Error('Missing task commands');
 console.log(JSON.stringify({passed:true,tasks:8,actions:commands.length,pageErrors:errors,actualCollection:false}));
 await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
