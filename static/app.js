// OCR Butterfly — Cosmic UI
// Complete rewrite for new cosmic dark theme + DeepSeek-OCR-2 model

// ============================================================
// Stardust Background
// ============================================================
(function initStardust() {
    const canvas = document.getElementById('starCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const stars = [];

    function regenerateStars() {
        stars.length = 0;
        const count = Math.floor((canvas.width * canvas.height) / 6500);
        for (let i = 0; i < count; i++) {
            stars.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                size: Math.random() * 1.5 + 0.45,
                opacity: Math.random() * 0.45 + 0.12,
                twinkleSpeed: Math.random() * 0.01 + 0.004,
                twinklePhase: Math.random() * Math.PI * 2,
            });
        }
    }

    const setSize = () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        regenerateStars();
    };
    setSize();

    function animate() {
        ctx.fillStyle = '#040916';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        stars.forEach(s => {
            if (!prefersReducedMotion) s.twinklePhase += s.twinkleSpeed;
            const t = prefersReducedMotion ? 1 : Math.sin(s.twinklePhase) * 0.24 + 0.72;
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.size, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(255,255,255,${s.opacity * t})`;
            ctx.fill();
            if (s.size > 1.5) {
                const g = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.size * 3);
                g.addColorStop(0, `rgba(255,255,255,${s.opacity * t * 0.2})`);
                g.addColorStop(1, 'rgba(255,255,255,0)');
                ctx.fillStyle = g;
                ctx.beginPath();
                ctx.arc(s.x, s.y, s.size * 3, 0, Math.PI * 2);
                ctx.fill();
            }
        });
        requestAnimationFrame(animate);
    }
    animate();
    window.addEventListener('resize', setSize);
})();

// ============================================================
// Card Glow Follow
// ============================================================
document.querySelectorAll('.card').forEach(card => {
    const glow = card.querySelector('.glow-follow');
    if (!glow) return;
    card.addEventListener('mousemove', e => {
        const r = card.getBoundingClientRect();
        const x = e.clientX - r.left;
        const y = e.clientY - r.top;
        glow.style.background = `radial-gradient(600px circle at ${x}px ${y}px, rgba(255,255,255,0.06), transparent 40%)`;
    });
});

// ============================================================
// Model Status Polling
// ============================================================
(function pollModelStatus() {
    const dot = document.getElementById('modelDot');
    const name = document.getElementById('modelName');

    async function check() {
        try {
            const res = await fetch('/api/status');
            const data = await res.json();
            if (data.model_healthy) {
                dot.classList.remove('offline');
                name.textContent = data.model_display_name || 'DeepSeek-OCR-2';
            } else {
                dot.classList.add('offline');
                name.textContent = 'Model offline';
            }
        } catch {
            dot.classList.add('offline');
            name.textContent = 'Disconnected';
        }
    }
    check();
    setInterval(check, 15000);
})();

// ============================================================
// Tab Switching
// ============================================================
document.querySelectorAll('.tab-nav .tab-btn').forEach(btn => {
    btn.addEventListener('click', function () {
        if (this.disabled) return;
        const tab = this.dataset.tab;

        document.querySelectorAll('.tab-nav .tab-btn').forEach(b => b.classList.remove('active'));
        this.classList.add('active');

        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        const content = document.getElementById(`tab-${tab}-content`);
        if (content) content.classList.add('active');

        if (tab === 'ocr') switchToOcrTab();
        else if (tab === 'video') switchToVideoTab();
        else if (tab === 'preprocess') switchToPreprocessTab();

        const titles = { ocr: 'Results', video: 'Results', preprocess: 'Preprocessing Results' };
        const titleEl = document.getElementById('resultTitle');
        if (titleEl) titleEl.textContent = titles[tab] || 'Results';
    });
});

// ============================================================
// Global State
// ============================================================
let selectedFile = null;
let currentTaskId = null;
let currentBatchIndex = 0;
let totalPages = 0;
let stopProcessing = false;
let pdfMode = 'batch';
let currentZoom = 1;
let currentMode = 'Document';
let currentSubcategory = 'Academic';
let currentComplexity = 'Medium';
let preprocessFiles = [];
let preprocessResults = [];
let currentPreprocessTaskId = null;
let videoFile = null;
let extractedFrames = [];
let currentVideoTaskId = null;
let imagePreprocessed = false;
let pdfPreprocessed = false;
let processedImageFile = null;
let processedPdfThumbnails = [];
let isPdfPreprocessMode = false;
let isVideoPreprocessMode = false;
let processedVideoFrames = [];

// ============================================================
// DOM Elements
// ============================================================
const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const preview = document.getElementById('preview');
const previewImage = document.getElementById('previewImage');
const clearBtn = document.getElementById('clearBtn');
const loading = document.getElementById('loading');
const loadingText = document.getElementById('loadingText');
const loadingSubtext = document.getElementById('loadingSubtext');
const errorDiv = document.getElementById('error');
const resultDiv = document.getElementById('result');
const successDiv = document.getElementById('success');
const copyBtn = document.getElementById('copyBtn');
const downloadBtn = document.getElementById('downloadBtn');

const modeSelect = document.getElementById('modeSelect');
const subcategorySelect = document.getElementById('subcategorySelect');
const complexitySlider = document.getElementById('complexitySlider');
const complexityLabel = document.getElementById('complexityLabel');
const configPanel = document.getElementById('configPanel');
const progressInfo = document.getElementById('progressInfo');
const pdfModeSection = document.getElementById('pdfModeSection');

const batchModeBtn = document.getElementById('batchModeBtn');
const singleModeBtn = document.getElementById('singleModeBtn');
const batchSettings = document.getElementById('batchSettings');
const singleSettings = document.getElementById('singleSettings');
const batchSizeSelect = document.getElementById('batchSize');
const customBatchInput = document.getElementById('customBatchSize');
const pageSelector = document.getElementById('pageSelector');
const thumbnailsContainer = document.getElementById('thumbnails');
const thumbnailsGrid = document.getElementById('thumbnailsGrid');

const processBatchBtn = document.getElementById('processBatchBtn');
const processSingleBtn = document.getElementById('processSingleBtn');
const processImageBtn = document.getElementById('processImageBtn');
const batchBtnText = document.getElementById('batchBtnText');
const singleBtnText = document.getElementById('singleBtnText');
const imageBtnText = document.getElementById('imageBtnText');

const batchControls = document.getElementById('batchControls');
const continueBtn = document.getElementById('continueBtn');
const stopBtn = document.getElementById('stopBtn');

const preprocessFileInput = document.getElementById('preprocessFileInput');
const preprocessDropZone = document.getElementById('preprocessDropZone');
const processPreprocessBtn = document.getElementById('processPreprocessBtn');
const downloadPreprocessBtn = document.getElementById('downloadPreprocessBtn');
const sendToOcrBtn = document.getElementById('sendToOcrBtn');
const preprocessPreview = document.getElementById('preprocessPreview');
const preprocessImages = document.getElementById('preprocessImages');
const preprocessProgressBar = document.getElementById('preprocessProgressBar');
const preprocessProgressText = document.getElementById('preprocessProgressText');

const videoFileInput = document.getElementById('videoFileInput');
const videoDropZone = document.getElementById('videoDropZone');
const extractFramesBtn = document.getElementById('extractFramesBtn');
const downloadFramesBtn = document.getElementById('downloadFramesBtn');
const sendFramesToOcrBtn = document.getElementById('sendFramesToOcrBtn');
const framesPreview = document.getElementById('framesPreview');
const framesGrid = document.getElementById('framesGrid');
const framesCount = document.getElementById('framesCount');
const selectAllFrames = document.getElementById('selectAllFrames');
const deselectAllFrames = document.getElementById('deselectAllFrames');

const imagePreprocessSection = document.getElementById('imagePreprocessSection');
const pdfPreprocessSection = document.getElementById('pdfPreprocessSection');
const skipImagePreprocessBtn = document.getElementById('skipImagePreprocessBtn');
const executeImagePreprocessBtn = document.getElementById('executeImagePreprocessBtn');
const skipPdfPreprocessBtn = document.getElementById('skipPdfPreprocessBtn');
const executePdfPreprocessBtn = document.getElementById('executePdfPreprocessBtn');

// ============================================================
// Mapping Tables
// ============================================================
const subcategoryMap = {
    Document: ['Academic', 'Business', 'Content', 'Table', 'Handwritten', 'Complex'],
    Scene: ['Street', 'Photo', 'Objects', 'Verification']
};
const complexityMap = { Tiny: 64, Small: 100, Medium: 256, Large: 400, Gundam: 800 };
const complexityNames = ['Tiny', 'Small', 'Medium', 'Large', 'Gundam'];

// ============================================================
// Config Selection
// ============================================================
modeSelect.addEventListener('change', e => { currentMode = e.target.value; updateSubcategoryOptions(); updateConfigDisplay(); });
subcategorySelect.addEventListener('change', e => { currentSubcategory = e.target.value; updateConfigDisplay(); });
complexitySlider.addEventListener('input', e => { currentComplexity = complexityNames[parseInt(e.target.value)]; updateConfigDisplay(); });

function updateSubcategoryOptions() {
    const cats = subcategoryMap[currentMode] || [];
    subcategorySelect.innerHTML = '';
    cats.forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c; subcategorySelect.appendChild(o); });
    currentSubcategory = cats[0] || 'Academic';
    subcategorySelect.value = currentSubcategory;
}

function updateConfigDisplay() {
    const tokens = complexityMap[currentComplexity] || 256;
    const hint = currentComplexity === 'Medium' ? ' (Recommended)' : '';
    complexityLabel.textContent = `${currentComplexity} — ${tokens} tokens${hint}`;
}

// ============================================================
// PDF Mode Toggle
// ============================================================
batchModeBtn.addEventListener('click', () => {
    pdfMode = 'batch';
    batchModeBtn.classList.add('active');
    singleModeBtn.classList.remove('active');
    batchSettings.classList.remove('hidden');
    singleSettings.classList.add('hidden');
    processBatchBtn.classList.remove('hidden');
    processSingleBtn.classList.add('hidden');
    resetResults();
});

singleModeBtn.addEventListener('click', () => {
    pdfMode = 'single';
    singleModeBtn.classList.add('active');
    batchModeBtn.classList.remove('active');
    batchSettings.classList.add('hidden');
    singleSettings.classList.remove('hidden');
    processBatchBtn.classList.add('hidden');
    processSingleBtn.classList.remove('hidden');
    resetResults();
});

// Batch size
batchSizeSelect.addEventListener('change', e => {
    customBatchInput.classList.toggle('hidden', e.target.value !== 'custom');
    if (e.target.value === 'custom') customBatchInput.focus();
});

customBatchInput.addEventListener('blur', () => {
    const val = parseInt(customBatchInput.value);
    const err = document.getElementById('batchValidationError');
    if (val < 1 || val > 50 || isNaN(val)) { err.textContent = 'Enter 1-50'; err.classList.remove('hidden'); customBatchInput.value = 5; }
    else err.classList.add('hidden');
});

// ============================================================
// File Upload (OCR Tab)
// ============================================================
if (dropZone && fileInput) {
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', e => { e.preventDefault(); dropZone.classList.remove('dragover'); handleFiles(e.dataTransfer.files); });
    fileInput.addEventListener('change', e => handleFiles(e.target.files));
}

clearBtn.addEventListener('click', () => {
    fileInput.value = '';
    selectedFile = null;
    resetState();
    configPanel.classList.add('hidden');
    imagePreprocessSection.classList.add('hidden');
    pdfPreprocessSection.classList.add('hidden');
    pdfModeSection.classList.add('hidden');
});

pageSelector.addEventListener('change', () => {
    const page = parseInt(pageSelector.value);
    if (page) selectPage(page);
});

function getCurrentBatchSize() {
    if (batchSizeSelect.value === 'custom') return Math.max(1, Math.min(parseInt(customBatchInput.value) || 1, 50));
    return parseInt(batchSizeSelect.value) || 2;
}

function selectPage(p) { pageSelector.value = p; loadPagePreview(p); }

// ============================================================
// Handle Uploaded Files
// ============================================================
function handleFiles(files) {
    if (!files || files.length === 0) return;
    const file = files[0];
    const fn = (file.name || '').toLowerCase();
    const isPdf = file.type === 'application/pdf' || fn.endsWith('.pdf');
    const isImage = file.type.startsWith('image/') || /\.(png|jpe?g)$/i.test(fn);
    if (!isImage && !isPdf) { showError('Only JPG, PNG or PDF supported'); return; }

    selectedFile = file;
    resetState();
    imagePreprocessed = false;
    pdfPreprocessed = false;
    processedImageFile = null;
    processedPdfThumbnails = [];
    isPdfPreprocessMode = false;

    if (isPdf) {
        pdfModeSection.classList.remove('hidden');
        pdfPreprocessSection.classList.remove('hidden');
        imagePreprocessSection.classList.add('hidden');
        configPanel.classList.add('hidden');
        processImageBtn.classList.add('hidden');
        processBatchBtn.classList.add('hidden');
        processSingleBtn.classList.add('hidden');
        thumbnailsContainer.classList.add('hidden');
    } else {
        pdfModeSection.classList.add('hidden');
        pdfPreprocessSection.classList.add('hidden');
        imagePreprocessSection.classList.remove('hidden');
        configPanel.classList.add('hidden');
        processBatchBtn.classList.add('hidden');
        processSingleBtn.classList.add('hidden');
        processImageBtn.classList.add('hidden');
        const reader = new FileReader();
        reader.onload = e => { previewImage.src = e.target.result; preview.classList.remove('hidden'); currentZoom = 1; previewImage.style.transform = 'scale(1)'; };
        reader.readAsDataURL(file);
    }
}

// ============================================================
// Image Preprocessing (OCR Tab inline)
// ============================================================
skipImagePreprocessBtn.addEventListener('click', () => {
    imagePreprocessSection.classList.add('hidden');
    configPanel.classList.remove('hidden');
    processImageBtn.classList.remove('hidden');
});

executeImagePreprocessBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    showLoading('Preprocessing image...');
    const formData = new FormData();
    formData.append('files', selectedFile);
    try {
        const uploadRes = await fetch('/api/preprocess/upload', { method: 'POST', body: formData });
        const uploadData = await uploadRes.json();
        if (!uploadData.success) throw new Error(uploadData.error || 'Upload failed');
        const taskId = uploadData.task_id;
        const settings = {
            auto_rotate: document.getElementById('imageAutoRotate').checked,
            enhance: document.getElementById('imageEnhance').checked,
            remove_shadows: document.getElementById('imageRemoveShadows').checked,
            binarize: document.getElementById('imageBinarize').checked,
            remove_bg: document.getElementById('imageRemoveBg').checked
        };
        const processRes = await fetch('/api/preprocess/process', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: taskId, settings })
        });
        const processData = await processRes.json();
        hideLoading();
        if (!processData.success) throw new Error('Processing failed');
        const completed = processData.results.find(r => r.status === 'completed');
        if (completed && completed.processed_thumb_b64) {
            previewImage.src = completed.processed_thumb_b64;
            preview.classList.remove('hidden');
        }
        if (completed && completed.processed_path) {
            const resp = await fetch(`/api/files/${completed.processed_path.split('/').slice(-2).join('/')}`);
            if (resp.ok) {
                const blob = await resp.blob();
                processedImageFile = new File([blob], selectedFile.name, { type: blob.type });
                imagePreprocessed = true;
            }
        }
        imagePreprocessSection.classList.add('hidden');
        configPanel.classList.remove('hidden');
        processImageBtn.classList.remove('hidden');
    } catch (err) {
        hideLoading();
        showError('Preprocessing failed: ' + err.message);
    }
});

// ============================================================
// PDF Preprocessing
// ============================================================
skipPdfPreprocessBtn.addEventListener('click', () => {
    pdfPreprocessSection.classList.add('hidden');
    configPanel.classList.remove('hidden');
    if (pdfMode === 'batch') processBatchBtn.classList.remove('hidden');
    else processSingleBtn.classList.remove('hidden');
    initPDFPages(selectedFile);
});

executePdfPreprocessBtn.addEventListener('click', () => {
    isPdfPreprocessMode = true;
    const preprocessTab = document.querySelector('.tab-btn[data-tab="preprocess"]');
    if (preprocessTab) { preprocessTab.disabled = false; preprocessTab.click(); }
    pdfPreprocessSection.classList.add('hidden');
    initPDFPages(selectedFile).then(() => {
        if (currentTaskId) extractPdfPagesForPreprocessing();
    });
});

async function extractPdfPagesForPreprocessing() {
    showLoading('Extracting PDF pages...');
    try {
        const res = await fetch('/api/pdf/extract-pages', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: currentTaskId })
        });
        const data = await res.json();
        hideLoading();
        if (!data.success) throw new Error(data.error || 'Extraction failed');
        preprocessFiles = [];
        for (const img of data.images) {
            try {
                const resp = await fetch(img.file_url);
                if (resp.ok) {
                    const blob = await resp.blob();
                    const file = new File([blob], img.filename, { type: 'image/png' });
                    preprocessFiles.push(file);
                }
            } catch (e) { console.warn('Failed to fetch extracted page:', e); }
        }
        if (preprocessImages) {
            preprocessImages.innerHTML = '';
            data.images.forEach(img => {
                if (img.thumb_b64) {
                    const div = document.createElement('div');
                    div.className = 'preprocess-preview-item';
                    div.innerHTML = `<img src="${img.thumb_b64}" alt="${img.filename}">`;
                    preprocessImages.appendChild(div);
                }
            });
        }
        if (preprocessPreview) preprocessPreview.classList.remove('hidden');
        const settingsEl = document.getElementById('preprocessSettings');
        if (settingsEl) settingsEl.classList.remove('hidden');
        if (processPreprocessBtn) { processPreprocessBtn.classList.remove('hidden'); processPreprocessBtn.disabled = false; }
    } catch (err) {
        hideLoading();
        showError('PDF extraction failed: ' + err.message);
    }
}

// ============================================================
// Initialize PDF
// ============================================================
async function initPDFPages(file) {
    showLoading('Loading PDF...');
    const formData = new FormData();
    formData.append('file', file);
    formData.append('content_type', currentMode);
    formData.append('subcategory', currentSubcategory);
    formData.append('complexity', currentComplexity);
    try {
        const res = await fetch('/api/pdf/init', { method: 'POST', body: formData });
        const data = await res.json();
        hideLoading();
        if (!data.success) { showError(data.error || 'Failed to load PDF'); return; }
        currentTaskId = data.task_id;
        totalPages = data.total_pages;
        pageSelector.innerHTML = '';
        for (let i = 1; i <= totalPages; i++) {
            const o = document.createElement('option');
            o.value = i; o.textContent = `Page ${i}`;
            pageSelector.appendChild(o);
        }
        if (thumbnailsGrid) {
            thumbnailsGrid.innerHTML = '';
            data.thumbnails.forEach((b64, idx) => {
                const div = document.createElement('div');
                div.className = 'thumb-item';
                div.innerHTML = `<img src="${b64}" alt="Page ${idx + 1}">`;
                div.addEventListener('click', () => {
                    document.querySelectorAll('.thumb-item').forEach(el => el.classList.remove('selected'));
                    div.classList.add('selected');
                    selectPage(idx + 1);
                });
                thumbnailsGrid.appendChild(div);
            });
        }
        thumbnailsContainer.classList.remove('hidden');
        selectPage(1);
        batchBtnText.innerText = `Start Batch (${totalPages} pages)`;
        processBatchBtn.disabled = false;
        processSingleBtn.disabled = false;
    } catch (err) { hideLoading(); showError('Load failed: ' + err.message); }
}

// ============================================================
// Page Preview
// ============================================================
async function loadProcessedVideoFramePreview(frameNumber) {
    if (!isVideoPreprocessMode || !processedVideoFrames || processedVideoFrames.length === 0) return;
    const f = processedVideoFrames.find(item => item.frame === frameNumber);
    if (f && f.processed_path) {
        try {
            let url = f.processed_path;
            if (url.includes('processed')) {
                const idx = url.indexOf('processed');
                const dir = url.substring(0, idx - 1).split('/').pop();
                url = `/api/files/${dir}/${url.substring(idx)}`;
            }
            const resp = await fetch(url);
            if (resp.ok) {
                const blob = await resp.blob();
                const reader = new FileReader();
                reader.onload = e => { previewImage.src = e.target.result; preview.classList.remove('hidden'); currentZoom = 1; previewImage.style.transform = 'scale(1)'; };
                reader.readAsDataURL(blob);
                return;
            }
        } catch (e) { console.warn('Cannot load processed frame:', e); }
    }
}

async function loadPagePreview(pageNumber) {
    if (isVideoPreprocessMode && pdfPreprocessed && processedVideoFrames && processedVideoFrames.length > 0) {
        await loadProcessedVideoFramePreview(pageNumber);
        return;
    }
    if (!currentTaskId) return;
    if (pdfPreprocessed && processedPdfThumbnails && processedPdfThumbnails.length > 0) {
        const pp = processedPdfThumbnails.find(item => item.page === pageNumber);
        if (pp && pp.processed_path) {
            try {
                let url = pp.processed_path;
                if (url.includes('processed')) {
                    const idx = url.indexOf('processed');
                    const dir = url.substring(0, idx - 1).split('/').pop();
                    url = `/api/files/${dir}/${url.substring(idx)}`;
                }
                const resp = await fetch(url);
                if (resp.ok) {
                    const blob = await resp.blob();
                    const reader = new FileReader();
                    reader.onload = e => { previewImage.src = e.target.result; preview.classList.remove('hidden'); currentZoom = 1; previewImage.style.transform = 'scale(1)'; };
                    reader.readAsDataURL(blob);
                    return;
                }
            } catch (e) { console.warn('Falling back to original preview:', e); }
        }
    }
    try {
        const res = await fetch('/api/pdf/preview-page', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: currentTaskId, page_number: pageNumber })
        });
        const data = await res.json();
        if (data.success) { previewImage.src = data.image; preview.classList.remove('hidden'); currentZoom = 1; previewImage.style.transform = 'scale(1)'; }
    } catch (err) { showError('Preview failed: ' + err.message); }
}

// ============================================================
// OCR Processing
// ============================================================
processImageBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    processImageBtn.disabled = true;
    imageBtnText.innerText = 'Processing...';
    showLoading('Running OCR...');
    resetResults();
    const formData = new FormData();
    const fileToUse = imagePreprocessed && processedImageFile ? processedImageFile : selectedFile;
    formData.append('file', fileToUse);
    formData.append('content_type', currentMode);
    formData.append('subcategory', currentSubcategory);
    formData.append('complexity', currentComplexity);
    try {
        const res = await fetch('/api/ocr', { method: 'POST', body: formData });
        const data = await res.json();
        hideLoading();
        if (data.success) displaySingleResult(`${currentMode} / ${currentSubcategory} / ${currentComplexity}`, data.text);
        else showError(data.error);
    } catch (err) { hideLoading(); showError(err.message); }
    finally { processImageBtn.disabled = false; imageBtnText.innerText = 'Start OCR'; }
});

// Batch
processBatchBtn.addEventListener('click', () => {
    currentBatchIndex = 0;
    stopProcessing = false;
    resetResults();
    processBatch();
});

async function processBatch() {
    const isVideoModePreprocessed = isVideoPreprocessMode && processedVideoFrames && processedVideoFrames.length > 0;
    const isVideoModeUnprocessed = isVideoPreprocessMode && preprocessFiles && preprocessFiles.length > 1 && (!processedVideoFrames || processedVideoFrames.length === 0);
    const isVideoMode = isVideoModePreprocessed || isVideoModeUnprocessed;
    if (stopProcessing || (!currentTaskId && !isVideoMode)) return;

    const batchSize = getCurrentBatchSize();
    const startPage = currentBatchIndex * batchSize + 1;
    const endPage = Math.min((currentBatchIndex + 1) * batchSize, totalPages);
    showLoading(isVideoMode ? `Processing frames ${startPage}-${endPage}...` : `Processing pages ${startPage}-${endPage}...`);

    try {
        if (isVideoModeUnprocessed) {
            const fd = new FormData();
            preprocessFiles.forEach(f => fd.append('files', f));
            const upRes = await fetch('/api/preprocess/upload', { method: 'POST', body: fd });
            if (!upRes.ok) throw new Error('Upload failed');
            const upData = await upRes.json();
            if (!upData.success) throw new Error('Upload failed');
            const paths = {};
            upData.images.forEach((img, i) => {
                const m = img.filename.match(/frame_(\d+)\.jpg/);
                paths[String(m ? parseInt(m[1]) : i + 1)] = img.raw_path;
            });
            processedVideoFrames = Object.keys(paths).map(n => ({ frame: parseInt(n), processed_path: paths[n] })).sort((a, b) => a.frame - b.frame);
        }

        const body = { task_id: currentTaskId || 'video_frames', batch_index: currentBatchIndex, batch_size: batchSize };
        if (isVideoMode && processedVideoFrames.length > 0) {
            const map = {};
            processedVideoFrames.forEach(i => { map[String(i.frame)] = i.processed_path; });
            body.processed_images = map;
            body.content_type = currentMode;
            body.subcategory = currentSubcategory;
            body.complexity = currentComplexity;
        } else if (pdfPreprocessed && processedPdfThumbnails.length > 0) {
            const map = {};
            processedPdfThumbnails.forEach(i => { map[String(i.page)] = i.processed_path; });
            body.processed_images = map;
        }

        const endpoint = isVideoMode ? '/api/video/process-batch' : '/api/pdf/process-batch';
        const res = await fetch(endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        if (!res.ok) { const e = await res.json().catch(() => ({})); hideLoading(); showError(e.error || 'Batch failed'); return; }
        const data = await res.json();
        hideLoading();
        if (!data.success) { showError(data.error || 'Batch failed'); return; }

        data.results.forEach(r => displayPageResult(r.page, r.text));
        const processed = data.processed_pages;
        document.getElementById('progressCurrent').textContent = processed;
        document.getElementById('progressTotal').textContent = totalPages;
        document.getElementById('progressBar').style.width = `${(processed / totalPages * 100).toFixed(1)}%`;
        document.getElementById('progressConfig').textContent = `${currentMode} / ${currentSubcategory} / ${currentComplexity}`;
        progressInfo.classList.remove('hidden');

        if (data.has_more) {
            currentBatchIndex = data.next_batch_index;
            batchControls.classList.remove('hidden');
            batchBtnText.innerText = `Continue (${processed}/${totalPages})`;
            downloadBtn.classList.remove('hidden');
        } else finishBatch();
    } catch (err) { hideLoading(); showError(err.message); }
}

// Single page
processSingleBtn.addEventListener('click', async () => {
    const page = parseInt(pageSelector.value);
    if (!page || !currentTaskId) return;
    processSingleBtn.disabled = true;
    singleBtnText.innerText = 'Recognizing...';
    showLoading(`Page ${page}...`);
    resetResults();
    try {
        const body = { task_id: currentTaskId, batch_index: page - 1, batch_size: 1 };
        if (pdfPreprocessed && processedPdfThumbnails.length > 0) {
            const map = {};
            processedPdfThumbnails.forEach(i => { map[String(i.page)] = i.processed_path; });
            body.processed_images = map;
        }
        const res = await fetch('/api/pdf/process-batch', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
        const data = await res.json();
        hideLoading();
        if (data.success && data.results[0]) {
            displayPageResult(page, data.results[0].text);
            successDiv.classList.remove('hidden');
            copyBtn.classList.remove('hidden');
            downloadBtn.classList.remove('hidden');
        } else showError(data.error || 'Failed');
    } catch (err) { hideLoading(); showError(err.message); }
    finally { processSingleBtn.disabled = false; singleBtnText.innerText = 'Recognize Page'; }
});

// ============================================================
// Display Results
// ============================================================
function displayPageResult(pageNum, text) {
    const existing = resultDiv.querySelector('.result-placeholder');
    if (existing) existing.remove();

    const div = document.createElement('div');
    div.className = 'page-result';
    div.innerHTML = `
        <div class="page-result-header">
            <h3>Page ${pageNum}</h3>
            <span class="char-count">${text.length} chars</span>
        </div>
        <pre>${escapeHtml(text)}</pre>`;
    resultDiv.appendChild(div);
}

function displaySingleResult(title, text) {
    resultDiv.innerHTML = `
        <div class="page-result">
            <div class="page-result-header"><h3>${escapeHtml(title)}</h3><span class="char-count">${text.length} chars</span></div>
            <pre>${escapeHtml(text)}</pre>
        </div>`;
    successDiv.classList.remove('hidden');
    copyBtn.classList.remove('hidden');
    downloadBtn.classList.remove('hidden');
}

function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function finishBatch() {
    successDiv.textContent = `Complete! ${totalPages} pages processed`;
    successDiv.classList.remove('hidden');
    copyBtn.classList.remove('hidden');
    downloadBtn.classList.remove('hidden');
    batchControls.classList.add('hidden');
    batchBtnText.innerText = 'Done';
    processBatchBtn.disabled = false;
    cleanupTask();
}

continueBtn.addEventListener('click', () => processBatch());
stopBtn.addEventListener('click', () => {
    stopProcessing = true;
    batchControls.classList.add('hidden');
    hideLoading();
    batchBtnText.innerText = 'Stopped';
    processBatchBtn.disabled = false;
    downloadBtn.classList.remove('hidden');
});

// ============================================================
// Copy / Download
// ============================================================
copyBtn.addEventListener('click', () => {
    const texts = Array.from(resultDiv.querySelectorAll('pre')).map(p => p.textContent);
    const full = texts.map((t, i) => `=== Page ${i + 1} ===\n${t}`).join('\n\n');
    navigator.clipboard.writeText(full).then(() => {
        const orig = copyBtn.textContent;
        copyBtn.textContent = 'Copied!';
        setTimeout(() => { copyBtn.textContent = orig; }, 2000);
    }).catch(e => showError('Copy failed: ' + e.message));
});

downloadBtn.addEventListener('click', () => {
    const texts = Array.from(resultDiv.querySelectorAll('pre')).map(p => p.textContent);
    const full = texts.map((t, i) => `=== Page ${i + 1} ===\n${t}`).join('\n\n');
    const blob = new Blob([full], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `OCR_Result_${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
    a.click();
    URL.revokeObjectURL(a.href);
});

// ============================================================
// State Management
// ============================================================
function resetState() {
    cleanupTask();
    currentTaskId = null;
    currentBatchIndex = 0;
    totalPages = 0;
    stopProcessing = false;
    thumbnailsGrid.innerHTML = '';
    thumbnailsContainer.classList.add('hidden');
    preview.classList.add('hidden');
    resultDiv.innerHTML = '<div class="result-placeholder">Upload a file to get started</div>';
    imagePreprocessSection.classList.add('hidden');
    pdfPreprocessSection.classList.add('hidden');
    pdfModeSection.classList.add('hidden');
    [successDiv, errorDiv, progressInfo, batchControls, copyBtn, downloadBtn].forEach(el => el.classList.add('hidden'));
    processBatchBtn.disabled = true;
    processSingleBtn.disabled = true;
    isPdfPreprocessMode = false;
}

function resetResults() {
    resultDiv.innerHTML = '<div class="result-placeholder">Upload a file to get started</div>';
    [successDiv, errorDiv, progressInfo, batchControls, copyBtn, downloadBtn].forEach(el => el.classList.add('hidden'));
}

function cleanupTask() {
    if (currentTaskId) {
        fetch('/api/pdf/cancel', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ task_id: currentTaskId }) }).catch(() => {});
    }
}

// ============================================================
// Tab State Switches
// ============================================================
function switchToOcrTab() {
    videoFile = null;
    extractedFrames = [];
    if (videoFileInput) videoFileInput.value = '';
    if (framesPreview) { framesPreview.classList.add('hidden'); if (framesGrid) framesGrid.innerHTML = ''; }
    if (downloadFramesBtn) downloadFramesBtn.classList.add('hidden');
    if (preprocessPreview) preprocessPreview.classList.add('hidden');

    const hasFile = (fileInput && fileInput.value) || selectedFile;
    if (!hasFile) {
        imagePreprocessSection.classList.add('hidden');
        pdfPreprocessSection.classList.add('hidden');
        pdfModeSection.classList.add('hidden');
        configPanel.classList.add('hidden');
        selectedFile = null;
    } else if (selectedFile) {
        if (selectedFile.type === 'application/pdf') {
            pdfModeSection.classList.remove('hidden');
            if (pdfPreprocessed && processedPdfThumbnails.length > 0) { thumbnailsContainer.classList.remove('hidden'); configPanel.classList.remove('hidden'); }
            else pdfPreprocessSection.classList.remove('hidden');
        } else {
            if (imagePreprocessed && processedImageFile) { configPanel.classList.remove('hidden'); processImageBtn.classList.remove('hidden'); }
            else imagePreprocessSection.classList.remove('hidden');
        }
    }
}

function switchToVideoTab() {
    if (preprocessPreview) preprocessPreview.classList.add('hidden');
}

function switchToPreprocessTab() {}

// ============================================================
// Video Tab
// ============================================================
if (videoDropZone && videoFileInput) {
    videoDropZone.addEventListener('click', () => videoFileInput.click());
    videoDropZone.addEventListener('dragover', e => { e.preventDefault(); videoDropZone.classList.add('dragover'); });
    videoDropZone.addEventListener('dragleave', () => videoDropZone.classList.remove('dragover'));
    videoDropZone.addEventListener('drop', e => { e.preventDefault(); videoDropZone.classList.remove('dragover'); handleVideoFiles(e.dataTransfer.files); });
    videoFileInput.addEventListener('change', e => handleVideoFiles(e.target.files));
}

function handleVideoFiles(files) {
    if (!files || files.length === 0) return;
    const file = files[0];
    const ext = (file.name || '').split('.').pop().toLowerCase();
    if (!['mp4', 'avi', 'mov', 'mkv', 'webm'].includes(ext)) { showError('Unsupported video format'); return; }
    videoFile = file;
    uploadVideo(file);
}

async function uploadVideo(file) {
    showLoading('Uploading video...');
    const fd = new FormData();
    fd.append('file', file);
    try {
        const res = await fetch('/api/video/upload', { method: 'POST', body: fd });
        const data = await res.json();
        hideLoading();
        if (!data.success) { showError(data.error || 'Upload failed'); return; }
        currentVideoTaskId = data.task_id;
        const info = data.video_info;
        document.getElementById('videoInfoContent').innerHTML =
            `<strong>${info.filename}</strong><br>Duration: ${info.duration.toFixed(1)}s | FPS: ${info.fps.toFixed(1)} | Frames: ${info.total_frames} | ${info.resolution}`;
        document.getElementById('videoInfo').classList.remove('hidden');
        document.getElementById('videoSettings').classList.remove('hidden');
        extractFramesBtn.classList.remove('hidden');
    } catch (err) { hideLoading(); showError(err.message); }
}

// Extraction method toggle
const extractionMethod = document.getElementById('extractionMethod');
if (extractionMethod) {
    extractionMethod.addEventListener('change', function () {
        document.getElementById('frameCountSetting').classList.add('hidden');
        document.getElementById('intervalSetting').classList.add('hidden');
        document.getElementById('sceneChangeSettings').classList.add('hidden');
        if (this.value === 'fixed_count') document.getElementById('frameCountSetting').classList.remove('hidden');
        else if (this.value === 'fixed_interval') document.getElementById('intervalSetting').classList.remove('hidden');
        else if (this.value === 'scene_change') document.getElementById('sceneChangeSettings').classList.remove('hidden');
    });
}

const sceneSensitivity = document.getElementById('sceneSensitivity');
if (sceneSensitivity) {
    sceneSensitivity.addEventListener('input', function () {
        document.getElementById('sensitivityLabel').textContent = this.value;
    });
}

extractFramesBtn.addEventListener('click', async () => {
    if (!currentVideoTaskId) return;
    showLoading('Extracting frames...');
    const settings = {
        method: document.getElementById('extractionMethod').value,
        interval: parseInt(document.getElementById('frameInterval').value) || 5,
        total_frames: parseInt(document.getElementById('totalFrames').value) || 1000,
        sensitivity: parseFloat(document.getElementById('sceneSensitivity').value) || 0.5,
        format: document.getElementById('outputFormat').value
    };
    try {
        const res = await fetch('/api/video/extract', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: currentVideoTaskId, settings })
        });
        const data = await res.json();
        hideLoading();
        if (!data.success) { showError(data.error || 'Extraction failed'); return; }
        extractedFrames = data.frames;
        renderFrames(data.frames);
        framesPreview.classList.remove('hidden');
        downloadFramesBtn.classList.remove('hidden');
        sendFramesToOcrBtn.classList.remove('hidden');
    } catch (err) { hideLoading(); showError(err.message); }
});

function renderFrames(frames) {
    framesGrid.innerHTML = '';
    frames.forEach(f => {
        const div = document.createElement('div');
        div.className = 'frame-item' + (f.selected ? ' selected' : '');
        div.innerHTML = `${f.thumb_b64 ? `<img src="${f.thumb_b64}" alt="Frame ${f.index}">` : ''}<span class="frame-num">#${f.index}</span>`;
        div.addEventListener('click', () => { div.classList.toggle('selected'); f.selected = !f.selected; updateFrameCount(); });
        framesGrid.appendChild(div);
    });
    updateFrameCount();
}

function updateFrameCount() {
    const sel = extractedFrames.filter(f => f.selected).length;
    framesCount.textContent = `${sel} / ${extractedFrames.length} selected`;
}

if (selectAllFrames) selectAllFrames.addEventListener('click', () => { extractedFrames.forEach(f => f.selected = true); renderFrames(extractedFrames); });
if (deselectAllFrames) deselectAllFrames.addEventListener('click', () => { extractedFrames.forEach(f => f.selected = false); renderFrames(extractedFrames); });

downloadFramesBtn.addEventListener('click', async () => {
    if (!currentVideoTaskId) return;
    const selected = extractedFrames.filter(f => f.selected).map(f => f.index);
    showLoading('Preparing download...');
    try {
        const res = await fetch('/api/video/download', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ task_id: currentVideoTaskId, selected_frames: selected })
        });
        const blob = await res.blob();
        hideLoading();
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `frames_${currentVideoTaskId}.zip`;
        a.click();
        URL.revokeObjectURL(a.href);
    } catch (err) { hideLoading(); showError(err.message); }
});

sendFramesToOcrBtn.addEventListener('click', () => {
    const selected = extractedFrames.filter(f => f.selected);
    if (selected.length === 0) { showError('No frames selected'); return; }
    isVideoPreprocessMode = true;
    preprocessFiles = [];
    processedVideoFrames = selected.map(f => ({ frame: f.index, processed_path: f.path, filename: `frame_${f.index}.jpg` }));
    totalPages = selected.length;
    currentBatchIndex = 0;
    stopProcessing = false;
    const ocrTab = document.querySelector('.tab-btn[data-tab="ocr"]');
    if (ocrTab) ocrTab.click();
    configPanel.classList.remove('hidden');
    processBatchBtn.classList.remove('hidden');
    batchBtnText.innerText = `Start Batch (${totalPages} frames)`;
    processBatchBtn.disabled = false;
    batchSettings.classList.remove('hidden');
    pdfModeSection.classList.remove('hidden');
    batchModeBtn.click();
});

// ============================================================
// Preprocess Tab (standalone)
// ============================================================
if (preprocessDropZone && preprocessFileInput) {
    preprocessDropZone.addEventListener('click', () => { if (!preprocessFileInput.disabled) preprocessFileInput.click(); });
    preprocessDropZone.addEventListener('dragover', e => { e.preventDefault(); preprocessDropZone.classList.add('dragover'); });
    preprocessDropZone.addEventListener('dragleave', () => preprocessDropZone.classList.remove('dragover'));
    preprocessDropZone.addEventListener('drop', e => { e.preventDefault(); preprocessDropZone.classList.remove('dragover'); if (!preprocessFileInput.disabled) handlePreprocessFiles(e.dataTransfer.files); });
    preprocessFileInput.addEventListener('change', e => handlePreprocessFiles(e.target.files));
}

function handlePreprocessFiles(files) {
    if (!files || files.length === 0) return;
    preprocessFiles = Array.from(files);
    const settingsEl = document.getElementById('preprocessSettings');
    if (settingsEl) settingsEl.classList.remove('hidden');
    if (processPreprocessBtn) { processPreprocessBtn.classList.remove('hidden'); processPreprocessBtn.disabled = false; }
    if (preprocessImages) {
        preprocessImages.innerHTML = '';
        preprocessFiles.forEach(f => {
            const reader = new FileReader();
            reader.onload = e => {
                const div = document.createElement('div');
                div.className = 'preprocess-preview-item';
                div.innerHTML = `<img src="${e.target.result}" alt="${f.name}">`;
                preprocessImages.appendChild(div);
            };
            reader.readAsDataURL(f);
        });
        if (preprocessPreview) preprocessPreview.classList.remove('hidden');
    }
}

if (processPreprocessBtn) {
    processPreprocessBtn.addEventListener('click', async () => {
        if (preprocessFiles.length === 0) return;
        showLoading('Processing images...');
        const fd = new FormData();
        preprocessFiles.forEach(f => fd.append('files', f));
        try {
            const upRes = await fetch('/api/preprocess/upload', { method: 'POST', body: fd });
            const upData = await upRes.json();
            if (!upData.success) throw new Error(upData.error || 'Upload failed');
            currentPreprocessTaskId = upData.task_id;
            const settings = {
                auto_rotate: document.getElementById('autoRotate').checked,
                enhance: document.getElementById('enhance').checked,
                remove_shadows: document.getElementById('removeShadows').checked,
                binarize: document.getElementById('binarize').checked,
                remove_bg: document.getElementById('removeBg').checked
            };
            const procRes = await fetch('/api/preprocess/process', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: currentPreprocessTaskId, settings })
            });
            const procData = await procRes.json();
            hideLoading();
            if (!procData.success) throw new Error('Processing failed');
            preprocessResults = procData.results;
            if (preprocessImages) {
                preprocessImages.innerHTML = '';
                procData.results.forEach(r => {
                    if (r.processed_thumb_b64) {
                        const div = document.createElement('div');
                        div.className = 'preprocess-preview-item';
                        div.innerHTML = `<img src="${r.processed_thumb_b64}" alt="${r.filename}">`;
                        preprocessImages.appendChild(div);
                    }
                });
            }
            if (preprocessPreview) preprocessPreview.classList.remove('hidden');
            if (downloadPreprocessBtn) { downloadPreprocessBtn.classList.remove('hidden'); downloadPreprocessBtn.disabled = false; }
            if (sendToOcrBtn) { sendToOcrBtn.classList.remove('hidden'); sendToOcrBtn.disabled = false; }
            if (isPdfPreprocessMode) {
                pdfPreprocessed = true;
                processedPdfThumbnails = procData.results.filter(r => r.status === 'completed').map((r, i) => ({
                    page: i + 1, processed_path: r.processed_path, processed_thumb_b64: r.processed_thumb_b64
                }));
            }
        } catch (err) { hideLoading(); showError('Processing failed: ' + err.message); }
    });
}

if (downloadPreprocessBtn) {
    downloadPreprocessBtn.addEventListener('click', async () => {
        if (!currentPreprocessTaskId) return;
        showLoading('Preparing download...');
        try {
            const res = await fetch('/api/preprocess/download', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ task_id: currentPreprocessTaskId })
            });
            const blob = await res.blob();
            hideLoading();
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `processed_${currentPreprocessTaskId}.zip`;
            a.click();
            URL.revokeObjectURL(a.href);
        } catch (err) { hideLoading(); showError(err.message); }
    });
}

if (sendToOcrBtn) {
    sendToOcrBtn.addEventListener('click', () => {
        if (isPdfPreprocessMode && pdfPreprocessed) {
            const ocrTab = document.querySelector('.tab-btn[data-tab="ocr"]');
            if (ocrTab) ocrTab.click();
            configPanel.classList.remove('hidden');
            if (pdfMode === 'batch') processBatchBtn.classList.remove('hidden');
            else processSingleBtn.classList.remove('hidden');
        }
    });
}

// Preset configs
const PREPROCESS_PRESETS = {
    remove_background_only: { auto_rotate: false, enhance: false, remove_shadows: false, binarize: false, remove_bg: true },
    photo_optimize: { auto_rotate: true, enhance: true, remove_shadows: true, binarize: false, remove_bg: false },
    scan_optimize: { auto_rotate: true, enhance: true, remove_shadows: true, binarize: true, remove_bg: false },
    enhance_blurry: { auto_rotate: false, enhance: true, remove_shadows: false, binarize: false, remove_bg: false }
};

window.applyPreprocessPreset = function (name) {
    const p = PREPROCESS_PRESETS[name];
    if (!p) return;
    document.getElementById('autoRotate').checked = p.auto_rotate;
    document.getElementById('enhance').checked = p.enhance;
    document.getElementById('removeShadows').checked = p.remove_shadows;
    document.getElementById('binarize').checked = p.binarize;
    document.getElementById('removeBg').checked = p.remove_bg;
    const sel = document.getElementById('preprocessPreset');
    if (sel) sel.value = name;
};

const preprocessPresetSelect = document.getElementById('preprocessPreset');
if (preprocessPresetSelect) {
    preprocessPresetSelect.addEventListener('change', function () {
        if (this.value !== 'custom') applyPreprocessPreset(this.value);
    });
}

// ============================================================
// Utility Functions
// ============================================================
function showLoading(text, sub) {
    loadingText.textContent = text || 'Processing...';
    if (loadingSubtext) loadingSubtext.textContent = sub || '';
    loading.classList.remove('hidden');
}

function hideLoading() { loading.classList.add('hidden'); }

function showError(msg) {
    errorDiv.textContent = msg;
    errorDiv.classList.remove('hidden');
    setTimeout(() => errorDiv.classList.add('hidden'), 8000);
}

window.zoomPreview = function (factor) {
    const img = document.getElementById('previewImage');
    if (!img) return;
    const m = img.style.transform.match(/scale\(([^)]+)\)/);
    const s = m ? parseFloat(m[1]) : 1;
    currentZoom = s * factor;
    img.style.transform = `scale(${currentZoom})`;
};
