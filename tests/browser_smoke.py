"""Browser acceptance against a running local server; no enrollment mutation."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright

ROOT=Path(__file__).resolve().parents[1]

def main():
    (ROOT/'output').mkdir(parents=True, exist_ok=True)
    results=[];errors=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,args=['--no-sandbox'])
        page=browser.new_page(viewport={'width':1440,'height':1100},device_scale_factor=1)
        page.on('pageerror',lambda err:errors.append(str(err)))
        page.goto('http://127.0.0.1:8000',wait_until='networkidle')
        page.wait_for_function("document.querySelector('#maximum').textContent !== '—'")
        assert not page.locator('#error').is_visible()
        assert page.locator('.airbag').count()==68
        page.screenshot(path=str(ROOT/'output/ui-desktop.png'),full_page=True)
        page.click('#play')
        page.wait_for_function("Number(document.querySelector('#timeline').value) >= 4")
        page.click('#play')
        frame=int(page.input_value('#timeline'))
        assert frame>=4
        page.click('[data-id="40"]')
        assert page.inner_text('#bag-name')=='气囊 40'
        page.locator('#timeline').fill('35');page.locator('#timeline').dispatch_event('change')
        page.wait_for_function("document.querySelector('#frame-badge').textContent === 'FRAME 035'")
        assert 'FRAME 035' in page.inner_text('#frame-badge')
        results.append('实时回放、暂停、时间轴、68气囊选择')
        page.click('[data-tab="posture"]')
        assert page.locator('.gallery-card').count()==4
        page.select_option('#gallery-person','dgs')
        page.locator('.gallery-card').first.click()
        assert page.locator('#image-dialog').is_visible()
        assert 'dgs' in page.inner_text('#image-title')
        page.click('#close-dialog')
        page.screenshot(path=str(ROOT/'output/ui-posture.png'),full_page=True)
        results.append('原始热力图组别切换与原图弹窗')
        page.click('[data-tab="regions"]')
        page.select_option('#region-person','hpy');page.select_option('#region-action','10')
        page.click('#region-load')
        page.wait_for_function("document.querySelector('#region-pose').textContent.includes('标注：左侧卧')")
        assert page.locator('.region-row').count()==5
        page.screenshot(path=str(ROOT/'output/ui-regions.png'),full_page=True)
        results.append('五区域预测、标注与睡姿切换')
        page.click('[data-tab="identity"]')
        page.select_option('#identity-person','dgs');page.click('#identity-load')
        page.wait_for_function("document.querySelector('#identity-info').textContent.includes('来源 dgs')")
        assert page.locator('#projection circle').count()>0
        page.select_option('#identity-person','SAI');page.click('#identity-load')
        page.wait_for_function("document.querySelector('#identity-info').textContent.includes('来源 SAI')")
        assert '原实验未注册用户' in page.inner_text('#identity-info')
        page.screenshot(path=str(ROOT/'output/ui-identity.png'),full_page=True)
        results.append('已注册/未注册用户识别与PCA散点图')
        page.click('[data-tab="evaluation"]')
        page.wait_for_function("document.querySelector('#evaluation-content').textContent.includes('98.42%')")
        assert '98.42%' in page.inner_text('#evaluation-content')
        assert '未达标' in page.inner_text('#evaluation-content')
        with page.expect_download() as download:
            page.click('a[download]')
        assert download.value.suggested_filename=='metrics.json'
        page.screenshot(path=str(ROOT/'output/ui-evaluation.png'),full_page=True)
        results.append('真实评估结果、未达标提示与报告下载')
        page.set_viewport_size({'width':390,'height':844})
        for tab in ['monitor','posture','regions','identity','evaluation']:
            page.click(f'[data-tab="{tab}"]')
            assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth'),tab
            page.screenshot(path=str(ROOT/f'output/ui-mobile-{tab}.png'),full_page=True)
        results.append('390px 手机尺寸五页面无横向溢出')
        assert not page.locator('#error').is_visible()
        assert not errors,errors
        browser.close()
    (ROOT/'output/browser_checks.json').write_text(json.dumps({'passed':True,'checks':results,'javascript_errors':errors},ensure_ascii=False,indent=2))
    print(json.dumps({'passed':True,'checks':results},ensure_ascii=False))

if __name__=='__main__':main()
