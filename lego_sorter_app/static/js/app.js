// LEGO Sorter Web Dashboard Controller
let currentTab = 'sorter';
let activeModalBinId = null;
let selectedCatalogParts = new Set();
let catalogPage = 1;
let currentCategory = 'All';
let currentSearch = '';
let autoScrollLogs = true;
let lastDetectionTimestamp = 0;
let statusPollTimer = null;
let logsPollTimer = null;

let allBins = [];
let allCategories = [];
let numConveyorSpots = 3;
let unassignedPage = 1;
let unassignedTotalPages = 1;
let unassignedCategory = 'All';
let unassignedSearch = '';

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  initTabs();
  initSettingsForm();
  loadBins();
  loadPresets();
  loadCategories();
  loadUnassignedBricks();
  
  // Polling loops
  statusPollTimer = setInterval(pollStatusAndMetrics, 800);
  logsPollTimer = setInterval(pollLogs, 1500);
});

// Tab Navigation
function initTabs() {
  const tabBtns = document.querySelectorAll('.nav-btn');
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      switchTab(target);
    });
  });
}

function switchTab(tabId) {
  currentTab = tabId;
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tabId));
  document.querySelectorAll('.tab-pane').forEach(p => p.classList.toggle('active', p.id === `tab-${tabId}`));
  
  if (tabId === 'bins') {
    loadBins();
    loadPresets();
    loadUnassignedBricks();
  } else if (tabId === 'logs') {
    pollLogs();
  } else if (tabId === 'settings') {
    loadSettings();
  }
}

// ----------------------------------------------------
// 1. Live Sorter & Conveyor Visualizer Polling
// ----------------------------------------------------
async function pollStatusAndMetrics() {
  try {
    const res = await fetch('/api/metrics');
    if (!res.ok) return;
    const data = await res.json();
    
    const liveFps = document.getElementById('live-fps-badge');
    const liveLat = document.getElementById('live-latency-badge');
    const liveSorted = document.getElementById('live-sorted-badge');
    if (liveFps) liveFps.textContent = `${data.fps} FPS`;
    if (liveLat) liveLat.textContent = `${data.inference_latency_ms} ms`;
    if (liveSorted) liveSorted.textContent = `${data.total_sorted} bricks`;
    
    if (document.getElementById('stat-fps')) {
      document.getElementById('stat-fps').textContent = data.fps;
      document.getElementById('stat-latency').textContent = `${data.inference_latency_ms} ms`;
      document.getElementById('stat-total').textContent = data.total_sorted;
      document.getElementById('stat-ppm').textContent = `${data.ppm} / min`;
      document.getElementById('stat-rejection').textContent = `${data.rejection_rate_percent}%`;
      document.getElementById('stat-uptime').textContent = formatSeconds(data.uptime_seconds);
    }

    if (data.recent_detections && data.recent_detections.length > 0) {
      const latest = data.recent_detections[0];
      if (latest.timestamp !== lastDetectionTimestamp) {
        lastDetectionTimestamp = latest.timestamp;
        updateLatestDetectionCard(latest);
        pulseChuteGate(latest.bin_id);
      }
      renderRecentDetectionsTable(data.recent_detections);
    }

    if (data.bin_distribution) {
      for (const [binId, count] of Object.entries(data.bin_distribution)) {
        const countElems = document.querySelectorAll(`.count-val-${binId}`);
        countElems.forEach(el => el.textContent = count);
      }
      renderDistributionTable(data.bin_distribution);
    }
  } catch (err) {
    console.error('Metrics poll error:', err);
  }
}

function updateLatestDetectionCard(det) {
  const thumb = document.getElementById('det-thumb');
  const title = document.getElementById('det-title');
  const meta = document.getElementById('det-meta');
  const binBadge = document.getElementById('det-bin-badge');
  
  if (thumb) thumb.src = det.thumbnail;
  if (title) title.textContent = `#${det.part_id} - ${det.name}`;
  if (meta) meta.textContent = `Category: ${det.category} | ${det.snapshots_taken} snapshots averaged | ${det.inference_time_ms}ms`;
  
  if (binBadge) {
    if (det.bin_id === -1) {
      binBadge.textContent = 'ROUTED TO: BIN -1 (END-OF-BELT REJECT)';
      binBadge.style.backgroundColor = '#ef4444';
    } else {
      const sideText = (det.side || 'top').toUpperCase();
      binBadge.textContent = `ROUTED TO: BIN ${det.bin_id} (Spot ${det.spot} ${sideText}, Row ${det.row}) [Delay: ${det.delay_ms}ms]`;
      binBadge.style.backgroundColor = '#10b981';
    }
  }

  const barsContainer = document.getElementById('det-top5-bars');
  if (barsContainer && det.top_5) {
    barsContainer.innerHTML = '';
    det.top_5.forEach((item, idx) => {
      const row = document.createElement('div');
      row.className = 'confidence-bar-row';
      const fillWidth = Math.max(2, item.confidence_percent);
      const color = idx === 0 ? '#3b82f6' : '#64748b';
      
      row.innerHTML = `
        <span class="bar-label" title="${item.part_id} - ${item.name}">#${item.part_id}</span>
        <div class="bar-track">
          <div class="bar-fill" style="width: ${fillWidth}%; background-color: ${color}"></div>
        </div>
        <span class="bar-value">${item.confidence_percent}%</span>
      `;
      barsContainer.appendChild(row);
    });
  }

  // Render actual series of camera snapshots sent to AI
  const cropsContainer = document.getElementById('det-captured-crops');
  const countBadge = document.getElementById('det-snaps-count-badge');
  if (cropsContainer) {
    cropsContainer.innerHTML = '';
    const crops = det.captured_crops || [];
    if (countBadge) countBadge.textContent = `${crops.length} frames`;

    if (crops.length === 0) {
      cropsContainer.innerHTML = `<div style="color: #64748b; font-size: 0.78rem; padding: 10px 0;">No snapshots recorded.</div>`;
    } else {
      crops.forEach((src, idx) => {
        const item = document.createElement('div');
        item.className = 'captured-crop-item';
        item.innerHTML = `
          <img src="${src}" alt="Frame ${idx + 1}">
          <span class="crop-idx-tag">Frame ${idx + 1}</span>
        `;
        cropsContainer.appendChild(item);
      });
    }
  }
}

function pulseChuteGate(binId) {
  const nodes = document.querySelectorAll(`.chute-node-${binId}`);
  nodes.forEach(node => {
    node.classList.add('gate-active');
    setTimeout(() => node.classList.remove('gate-active'), 900);
  });
}

function formatSeconds(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}m ${s}s`;
}

// ----------------------------------------------------
// 2. Two-Sided Conveyor & Vertical Chute Layout Schematics
// ----------------------------------------------------
async function loadBins() {
  try {
    const res = await fetch('/api/bins');
    const data = await res.json();
    allBins = data.bins;
    numConveyorSpots = data.num_conveyor_spots || 3;
    
    renderTwoSidedConveyorVisualizer('live-conveyor-schematic', allBins, false);
    renderTwoSidedConveyorVisualizer('config-conveyor-schematic', allBins, true);
    renderBinConfigCards(allBins);
  } catch (err) {
    console.error('Error loading bins:', err);
  }
}

function renderTwoSidedConveyorVisualizer(containerId, bins, isDraggable) {
  const container = document.getElementById(containerId);
  if (!container) return;
  container.innerHTML = '';

  const physicalBins = bins.filter(b => b.id >= 0);
  const rejectBin = bins.find(b => b.id === -1);

  // Compute maximum spot
  let maxSpot = Math.max(numConveyorSpots - 1, 0);
  physicalBins.forEach(b => {
    if (b.spot !== undefined && b.spot > maxSpot) maxSpot = b.spot;
  });

  const wrapper = document.createElement('div');
  wrapper.className = 'conveyor-two-sided-wrapper';

  // --- 1. TOP SIDE CHUTES ---
  const topSideContainer = document.createElement('div');
  topSideContainer.innerHTML = `<div class="conveyor-side-label">&#x25B2; TOP / NORTH SIDE CHUTES (Gates Diverting Upward)</div>`;
  const topRow = document.createElement('div');
  topRow.className = 'conveyor-side-row top-side';

  for (let s = 0; s <= maxSpot; s++) {
    const spotBins = physicalBins.filter(b => (b.spot === s && (b.side || 'top') === 'top')).sort((a, b) => (a.row || 0) - (b.row || 0));
    topRow.appendChild(createChuteSpotColumn(s, 'top', spotBins, isDraggable));
  }
  topSideContainer.appendChild(topRow);
  wrapper.appendChild(topSideContainer);

  // --- 2. CENTER CONVEYOR BELT STRIP & REJECT BIN ---
  const centerBeltRow = document.createElement('div');
  centerBeltRow.className = 'conveyor-center-belt-row';

  const beltStrip = document.createElement('div');
  beltStrip.className = 'conveyor-belt-strip';
  beltStrip.innerHTML = `<span class="conveyor-belt-arrow">&rarr;&rarr;&rarr; MAIN CONVEYOR BELT DIRECTION &rarr;&rarr;&rarr;</span>`;
  centerBeltRow.appendChild(beltStrip);

  // End Reject Bin -1 attached to the end of the belt
  const rejectCol = document.createElement('div');
  rejectCol.className = 'conveyor-reject-column';
  rejectCol.innerHTML = `
    <div class="chute-bin-node reject-node chute-node--1">
      <div class="bin-node-id" style="color: #f87171">Bin -1 (End Reject)</div>
      <div class="bin-node-count count-val--1">${rejectBin ? (rejectBin.count || 0) : 0}</div>
      <div class="bin-node-meta">Unassigned / Low Conf</div>
    </div>
  `;
  centerBeltRow.appendChild(rejectCol);
  wrapper.appendChild(centerBeltRow);

  // --- 3. BOTTOM SIDE CHUTES ---
  const bottomSideContainer = document.createElement('div');
  bottomSideContainer.innerHTML = `<div class="conveyor-side-label">&#x25BC; BOTTOM / SOUTH SIDE CHUTES (Gates Diverting Downward)</div>`;
  const bottomRow = document.createElement('div');
  bottomRow.className = 'conveyor-side-row bottom-side';

  for (let s = 0; s <= maxSpot; s++) {
    const spotBins = physicalBins.filter(b => (b.spot === s && (b.side || 'top') === 'bottom')).sort((a, b) => (a.row || 0) - (b.row || 0));
    bottomRow.appendChild(createChuteSpotColumn(s, 'bottom', spotBins, isDraggable));
  }
  bottomSideContainer.appendChild(bottomRow);
  wrapper.appendChild(bottomSideContainer);

  container.appendChild(wrapper);
}

function createChuteSpotColumn(spot, side, spotBins, isDraggable) {
  const spotCol = document.createElement('div');
  spotCol.className = 'chute-spot-column';
  spotCol.dataset.spot = spot;
  spotCol.dataset.side = side;

  // Drop target for dragging bin to this spot & side
  if (isDraggable) {
    spotCol.addEventListener('dragover', (e) => {
      e.preventDefault();
      spotCol.classList.add('drag-target-hover');
    });
    spotCol.addEventListener('dragleave', () => {
      spotCol.classList.remove('drag-target-hover');
    });
    spotCol.addEventListener('drop', (e) => {
      e.preventDefault();
      spotCol.classList.remove('drag-target-hover');
      const binIdStr = e.dataTransfer.getData('text/bin-id');
      if (binIdStr) {
        const binId = parseInt(binIdStr);
        moveBinToSpotSide(binId, spot, side);
      }
    });
  }

  spotCol.innerHTML = `
    <div class="chute-spot-header">
      <span>SPOT ${spot} (${side.toUpperCase()})</span>
      <span class="master-gate-badge">Gate ${spot}</span>
    </div>
  `;

  const rowsContainer = document.createElement('div');
  rowsContainer.className = 'chute-rows-container';

  if (spotBins.length === 0) {
    rowsContainer.innerHTML = `<div style="color: #64748b; font-size: 0.72rem; text-align: center; padding: 10px 4px;">Empty Chute<br>(Drag bin here)</div>`;
  } else {
    spotBins.forEach((b, rIdx) => {
      const binNode = document.createElement('div');
      binNode.className = `chute-bin-node chute-node-${b.id}`;
      binNode.style.borderTopColor = b.color || '#3b82f6';
      binNode.dataset.binId = String(b.id);

      if (isDraggable) {
        binNode.draggable = true;
        binNode.addEventListener('dragstart', (e) => {
          e.dataTransfer.setData('text/bin-id', String(b.id));
          binNode.classList.add('dragging');
        });
        binNode.addEventListener('dragend', () => {
          binNode.classList.remove('dragging');
        });
      }

      // Allow dropping brick parts directly onto this schematic chute node!
      binNode.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        binNode.classList.add('drag-brick-hover');
      });
      binNode.addEventListener('dragleave', () => {
        binNode.classList.remove('drag-brick-hover');
      });
      binNode.addEventListener('drop', async (e) => {
        e.preventDefault();
        e.stopPropagation();
        binNode.classList.remove('drag-brick-hover');

        const partId = e.dataTransfer.getData('text/lego-part-id') || e.dataTransfer.getData('text/plain');
        const movingBinId = e.dataTransfer.getData('text/bin-id');

        if (partId && !movingBinId) {
          await assignPartToBinDirect(b.id, partId);
          pulseChuteGate(b.id);
        }
      });

      binNode.innerHTML = `
        <div class="bin-node-id" style="color: ${b.color || '#fff'}">Bin ${b.id} (Tier ${b.row || rIdx})</div>
        <div class="bin-node-count count-val-${b.id}">${b.count || 0}</div>
        <div class="bin-node-meta">Delay: ${b.delay_ms || 0}ms</div>
      `;
      rowsContainer.appendChild(binNode);
    });
  }

  spotCol.appendChild(rowsContainer);
  return spotCol;
}

async function moveBinToSpotSide(binId, targetSpot, targetSide) {
  const bin = allBins.find(b => b.id === binId);
  if (!bin) return;

  const existingInSpotSide = allBins.filter(b => b.id >= 0 && b.spot === targetSpot && (b.side || 'top') === targetSide && b.id !== binId);
  const targetRow = existingInSpotSide.length;

  bin.spot = targetSpot;
  bin.side = targetSide;
  bin.row = targetRow;
  bin.delay_ms = (targetSpot + 1) * 350;

  try {
    await fetch('/api/bins/update_bin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: bin.id,
        spot: bin.spot,
        side: bin.side,
        row: bin.row,
        delay_ms: bin.delay_ms
      })
    });
    loadBins();
  } catch (err) {
    console.error('Error moving bin:', err);
  }
}

// ----------------------------------------------------
// 3. Spots Add / Remove Controls
// ----------------------------------------------------
async function addConveyorSpot() {
  try {
    const res = await fetch('/api/spots/add', { method: 'POST' });
    const data = await res.json();
    numConveyorSpots = data.num_conveyor_spots;
    loadBins();
  } catch (err) {
    alert(`Error adding spot: ${err}`);
  }
}

async function removeConveyorSpot() {
  if (numConveyorSpots <= 1) {
    alert('At least 1 conveyor spot is required.');
    return;
  }
  if (!confirm(`Are you sure you want to remove Spot ${numConveyorSpots - 1}? Any bins in this spot will be moved to Spot ${numConveyorSpots - 2}.`)) return;

  try {
    const res = await fetch('/api/spots/remove', { method: 'POST' });
    const data = await res.json();
    numConveyorSpots = data.num_conveyor_spots;
    loadBins();
  } catch (err) {
    alert(`Error removing spot: ${err}`);
  }
}

// ----------------------------------------------------
// 4. Presets Management (Save, Load, Delete)
// ----------------------------------------------------
async function loadPresets() {
  try {
    const res = await fetch('/api/presets');
    const data = await res.json();
    const select = document.getElementById('preset-select');
    if (!select) return;

    select.innerHTML = '';
    if (data.presets.length === 0) {
      select.innerHTML = '<option value="">(No Presets Saved)</option>';
      return;
    }

    data.presets.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p;
      opt.textContent = p;
      if (p === data.active) opt.selected = true;
      select.appendChild(opt);
    });
  } catch (err) {
    console.error('Error loading presets:', err);
  }
}

async function saveNewPreset() {
  const name = prompt('Enter a name for this bin preset layout:');
  if (!name || !name.trim()) return;

  try {
    const res = await fetch('/api/presets/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name.trim() })
    });
    if (res.ok) {
      alert(`Preset '${name.trim()}' saved successfully!`);
      loadPresets();
    }
  } catch (err) {
    alert(`Error saving preset: ${err}`);
  }
}

async function loadSelectedPreset() {
  const select = document.getElementById('preset-select');
  const name = select ? select.value : '';
  if (!name) {
    alert('Please select a preset to load.');
    return;
  }

  if (!confirm(`Load preset '${name}'? This will replace your current bin assignments with the preset.`)) return;

  try {
    const res = await fetch('/api/presets/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name })
    });
    if (res.ok) {
      loadBins();
      loadUnassignedBricks();
    }
  } catch (err) {
    alert(`Error loading preset: ${err}`);
  }
}

async function deleteSelectedPreset() {
  const select = document.getElementById('preset-select');
  const name = select ? select.value : '';
  if (!name) return;

  if (!confirm(`Are you sure you want to permanently delete preset '${name}'?`)) return;

  try {
    await fetch(`/api/presets/${encodeURIComponent(name)}`, { method: 'DELETE' });
    loadPresets();
  } catch (err) {
    alert(`Error deleting preset: ${err}`);
  }
}

// ----------------------------------------------------
// 5. Bin Config Cards with Clear, Delete & Settings
// ----------------------------------------------------
const expandedBins = new Set();

function toggleBinCard(binId) {
  if (expandedBins.has(binId)) {
    expandedBins.delete(binId);
  } else {
    expandedBins.add(binId);
  }
  const card = document.getElementById(`bin-card-${binId}`);
  const btn = document.getElementById(`bin-expand-btn-${binId}`);
  if (card) {
    const isExpanded = expandedBins.has(binId);
    card.classList.toggle('expanded', isExpanded);
    if (btn) {
      const count = btn.getAttribute('data-count') || '0';
      btn.innerHTML = isExpanded ? `&#9662; Hide (${count})` : `&#9656; Parts (${count})`;
    }
  }
}

function renderBinConfigCards(bins) {
  const container = document.getElementById('bins-list-grid');
  if (!container) return;
  container.innerHTML = '';

  const physical = bins.filter(b => b.id >= 0).sort((a, b) => a.id - b.id);
  const reject = bins.find(b => b.id === -1);

  physical.forEach(bin => {
    const isExpanded = expandedBins.has(bin.id);
    const partsCount = (bin.assigned_parts || []).length;
    const card = document.createElement('div');
    card.className = `bin-config-card${isExpanded ? ' expanded' : ''}`;
    card.id = `bin-card-${bin.id}`;
    card.style.borderTop = `3px solid ${bin.color || '#3b82f6'}`;

    card.addEventListener('dragover', (e) => {
      e.preventDefault();
      card.classList.add('drag-brick-hover');
    });
    card.addEventListener('dragleave', () => {
      card.classList.remove('drag-brick-hover');
    });
    card.addEventListener('drop', async (e) => {
      e.preventDefault();
      card.classList.remove('drag-brick-hover');
      const partId = e.dataTransfer.getData('text/lego-part-id') || e.dataTransfer.getData('text/plain');
      if (partId) {
        await assignPartToBinDirect(bin.id, partId);
      }
    });

    let chipsHtml = '';
    if (bin.assigned_parts && bin.assigned_parts.length > 0) {
      bin.assigned_parts.forEach(pid => {
        chipsHtml += `
          <div class="part-chip">
            <img src="/static/thumbnails/${pid}.png" onerror="this.src='/static/thumbnails/3001.png'" alt="${pid}">
            <span>#${pid}</span>
            <span class="part-chip-remove" onclick="removePartFromBin(${bin.id}, '${pid}')">&times;</span>
          </div>
        `;
      });
    } else {
      chipsHtml = `<div style="color: #64748b; font-size: 0.76rem; padding: 4px;">No parts assigned. Drag bricks onto this card or click '+ Assign'.</div>`;
    }

    const sideLabel = (bin.side || 'top') === 'top' ? 'Top' : 'Bottom';
    card.innerHTML = `
      <div class="bin-header">
        <input type="text" class="bin-name-input" value="${bin.name}" onchange="updateBinName(${bin.id}, this.value)">
        <span class="badge" style="background: ${bin.color}22; color: ${bin.color}">Bin ${bin.id}</span>
        <button class="btn btn-sm btn-danger" style="padding: 2px 6px;" onclick="deleteBin(${bin.id})" title="Delete Bin">&times;</button>
      </div>

      <div class="bin-compact-summary">
        <span>Spot ${bin.spot || 0} &bull; ${sideLabel} &bull; T${bin.row || 0} &bull; ${bin.delay_ms || 400}ms</span>
        <div style="display: flex; gap: 4px; align-items: center;">
          <button class="btn btn-primary btn-sm" style="padding: 2px 7px; font-size: 0.72rem;" onclick="openPartPickerModal(${bin.id})">+ Assign</button>
          <button class="bin-expand-btn" id="bin-expand-btn-${bin.id}" data-count="${partsCount}" onclick="toggleBinCard(${bin.id})">
            ${isExpanded ? `&#9662; Hide (${partsCount})` : `&#9656; Parts (${partsCount})`}
          </button>
        </div>
      </div>

      <div class="bin-details-collapsible">
        <div class="bin-hardware-row">
          <div>
            <label>Spot (0-${numConveyorSpots - 1}):</label>
            <input type="number" min="0" max="${Math.max(10, numConveyorSpots)}" value="${bin.spot || 0}" onchange="updateBinProperty(${bin.id}, 'spot', parseInt(this.value))">
          </div>
          <div>
            <label>Belt Side:</label>
            <select onchange="updateBinProperty(${bin.id}, 'side', this.value)">
              <option value="top" ${(bin.side || 'top') === 'top' ? 'selected' : ''}>Top / North</option>
              <option value="bottom" ${(bin.side || 'top') === 'bottom' ? 'selected' : ''}>Bottom / South</option>
            </select>
          </div>
          <div>
            <label>Chute Tier:</label>
            <input type="number" min="0" max="10" value="${bin.row || 0}" onchange="updateBinProperty(${bin.id}, 'row', parseInt(this.value))">
          </div>
          <div>
            <label>Delay (ms):</label>
            <input type="number" min="0" step="25" value="${bin.delay_ms || 400}" onchange="updateBinProperty(${bin.id}, 'delay_ms', parseInt(this.value))">
          </div>
        </div>

        <div class="assigned-parts-grid" id="assigned-grid-${bin.id}">
          ${chipsHtml}
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 6px; gap: 6px; flex-wrap: wrap;">
          <button class="btn btn-warning btn-sm" onclick="clearBin(${bin.id})" title="Remove all parts from this bin">
            Clear Parts
          </button>
          <button class="btn btn-sm" onclick="testTriggerBin(${bin.id})">
            Test Gate
          </button>
        </div>
      </div>
    `;
    container.appendChild(card);
  });

  if (reject) {
    const rCard = document.createElement('div');
    rCard.className = 'bin-config-card';
    rCard.style.borderTop = '3px solid #ef4444';
    rCard.style.background = '#1c1518';
    rCard.innerHTML = `
      <div class="bin-header">
        <span style="font-weight: 700; color: #f87171; font-size: 0.88rem;">Bin -1: Reject / Overflow</span>
        <span class="badge" style="background: #ef444422; color: #f87171;">Protected</span>
      </div>
      <div class="bin-compact-summary">
        <span>End-of-belt (&lt;35% conf or unassigned)</span>
        <button class="btn btn-sm" style="padding: 2px 7px; font-size: 0.72rem;" onclick="testTriggerBin(-1)">Test</button>
      </div>
    `;
    container.appendChild(rCard);
  }
}

async function addBin() {
  const name = prompt('Enter a name for the new bin (or leave blank for default):');
  try {
    const res = await fetch('/api/bins/add', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name: name || null })
    });
    if (res.ok) {
      loadBins();
    }
  } catch (err) {
    alert(`Failed to add bin: ${err}`);
  }
}

async function deleteBin(binId) {
  if (!confirm(`Are you sure you want to permanently delete Bin ${binId}? Any assigned parts will return to unassigned.`)) return;
  try {
    const res = await fetch(`/api/bins/${binId}`, { method: 'DELETE' });
    if (res.ok) {
      loadBins();
      loadUnassignedBricks();
    }
  } catch (err) {
    alert(`Failed to delete bin: ${err}`);
  }
}

async function clearBin(binId) {
  if (!confirm(`Are you sure you want to clear all parts from Bin ${binId}?\nAll assigned bricks will return to the unassigned list.`)) return;
  try {
    const res = await fetch(`/api/bins/${binId}/clear`, { method: 'POST' });
    if (res.ok) {
      loadBins();
      loadUnassignedBricks();
    }
  } catch (err) {
    alert(`Error clearing bin: ${err}`);
  }
}

async function clearAllBins() {
  if (!confirm('Are you sure you want to clear all assigned parts from ALL bins?\nAll bricks will return to the unassigned list.')) return;
  try {
    const res = await fetch('/api/bins/clear_all', { method: 'POST' });
    if (res.ok) {
      loadBins();
      loadUnassignedBricks();
    }
  } catch (err) {
    alert(`Error clearing all bins: ${err}`);
  }
}

async function removeAllBins() {
  if (!confirm('Are you sure you want to DELETE ALL physical bins?\nThis will remove all bins except the end-of-belt reject bin.')) return;
  try {
    const res = await fetch('/api/bins/remove_all', { method: 'POST' });
    if (res.ok) {
      loadBins();
      loadUnassignedBricks();
    }
  } catch (err) {
    alert(`Error removing all bins: ${err}`);
  }
}

async function updateBinProperty(binId, key, value) {
  const bin = allBins.find(b => b.id === binId);
  if (!bin) return;
  bin[key] = value;

  try {
    await fetch('/api/bins/update_bin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: binId, [key]: value })
    });
    if (key === 'spot' || key === 'side' || key === 'row') {
      renderTwoSidedConveyorVisualizer('live-conveyor-schematic', allBins, false);
      renderTwoSidedConveyorVisualizer('config-conveyor-schematic', allBins, true);
    }
  } catch (err) {
    console.error('Error updating bin property:', err);
  }
}

function updateBinName(binId, newName) {
  updateBinProperty(binId, 'name', newName);
}

async function removePartFromBin(binId, partId) {
  try {
    await fetch('/api/bins/remove_part', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bin_id: binId, part_id: partId })
    });
    loadBins();
    loadUnassignedBricks();
  } catch (err) {
    console.error('Error removing part:', err);
  }
}

async function assignPartToBinDirect(binId, partId) {
  try {
    await fetch('/api/bins/assign_part', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bin_id: binId, part_id: partId })
    });
    loadBins();
    loadUnassignedBricks();
  } catch (err) {
    console.error('Error assigning part:', err);
  }
}

async function testTriggerBin(binId) {
  try {
    const res = await fetch('/api/test/trigger_bin', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bin_id: binId })
    });
    pulseChuteGate(binId);
  } catch (err) {
    alert(`Failed to trigger gate: ${err}`);
  }
}

// ----------------------------------------------------
// 6. Drag-and-Drop Unassigned Bricks Panel
// ----------------------------------------------------
async function populateUnassignedCategories() {
  const select = document.getElementById('unassigned-cat-select');
  if (!select || select.children.length > 0) return;

  if (allCategories.length === 0) {
    await loadCategories();
  }

  select.innerHTML = '';
  allCategories.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c.name;
    opt.textContent = `${c.name} (${c.count})`;
    if (c.name === unassignedCategory) opt.selected = true;
    select.appendChild(opt);
  });
}

async function loadUnassignedBricks() {
  const container = document.getElementById('unassigned-bricks-list');
  if (!container) return;

  await populateUnassignedCategories();

  try {
    const query = new URLSearchParams({
      category: unassignedCategory,
      search: unassignedSearch,
      page: unassignedPage,
      limit: 120
    });
    const res = await fetch(`/api/catalog/unassigned?${query}`);
    const data = await res.json();

    unassignedTotalPages = data.pages || 1;

    const titleEl = document.getElementById('unassigned-count-title');
    if (titleEl) titleEl.textContent = `Unassigned Bricks (${data.total})`;

    const pageInfo = document.getElementById('unassigned-page-info');
    if (pageInfo) pageInfo.textContent = `Page ${data.page} of ${data.pages}`;

    const prevBtn = document.getElementById('unassigned-prev-btn');
    const nextBtn = document.getElementById('unassigned-next-btn');
    if (prevBtn) prevBtn.disabled = (data.page <= 1);
    if (nextBtn) nextBtn.disabled = (data.page >= data.pages);

    container.innerHTML = '';
    if (data.items.length === 0) {
      container.innerHTML = `<div style="grid-column: 1/-1; color: #64748b; font-size: 0.78rem; padding: 16px; text-align: center;">No bricks match your filter.</div>`;
      return;
    }

    data.items.forEach(item => {
      const chip = document.createElement('div');
      chip.className = 'unassigned-brick-chip';
      chip.draggable = true;
      chip.title = `${item.id} - ${item.name} (${item.category})`;
      chip.innerHTML = `
        <img src="${item.thumbnail}" onerror="this.src='/static/thumbnails/3001.png'" alt="${item.id}">
        <div class="chip-id">#${item.id}</div>
      `;

      chip.addEventListener('dragstart', (e) => {
        e.dataTransfer.setData('text/plain', item.id);
        e.dataTransfer.setData('text/lego-part-id', item.id);
        chip.style.opacity = '0.5';
      });

      chip.addEventListener('dragend', () => {
        chip.style.opacity = '1';
      });

      container.appendChild(chip);
    });
  } catch (err) {
    console.error('Error loading unassigned bricks:', err);
  }
}

function prevUnassignedPage() {
  if (unassignedPage > 1) {
    unassignedPage--;
    loadUnassignedBricks();
  }
}

function nextUnassignedPage() {
  if (unassignedPage < unassignedTotalPages) {
    unassignedPage++;
    loadUnassignedBricks();
  }
}

function changeUnassignedCategory(cat) {
  unassignedCategory = cat;
  unassignedPage = 1;
  loadUnassignedBricks();
}

document.getElementById('unassigned-search')?.addEventListener('input', (e) => {
  unassignedSearch = e.target.value;
  unassignedPage = 1;
  loadUnassignedBricks();
});

// ----------------------------------------------------
// 7. Part Picker Modal with Category Pills
// ----------------------------------------------------
function openPartPickerModal(binId) {
  activeModalBinId = binId;
  const bin = allBins.find(b => b.id === binId);
  document.getElementById('modal-bin-title').textContent = `Assign Parts to Bin ${binId}: ${bin ? bin.name : ''}`;
  
  selectedCatalogParts = new Set(bin ? (bin.assigned_parts || []) : []);
  catalogPage = 1;
  document.getElementById('modal-search').value = '';
  currentSearch = '';
  currentCategory = 'All';
  
  document.getElementById('part-picker-modal').classList.add('active');
  renderModalCategoryPills();
  fetchCatalog();
}

function closePartPickerModal() {
  document.getElementById('part-picker-modal').classList.remove('active');
  activeModalBinId = null;
}

async function loadCategories() {
  try {
    const res = await fetch('/api/catalog/categories');
    allCategories = await res.json();
  } catch (err) {
    console.error('Error loading categories:', err);
  }
}

function renderModalCategoryPills() {
  const container = document.getElementById('modal-category-pills');
  if (!container) return;
  container.innerHTML = '';

  allCategories.forEach(c => {
    const btn = document.createElement('button');
    btn.className = `cat-pill-btn ${c.name === currentCategory ? 'active' : ''}`;
    btn.textContent = `${c.name} (${c.count})`;
    btn.addEventListener('click', () => {
      currentCategory = c.name;
      catalogPage = 1;
      renderModalCategoryPills();
      fetchCatalog();
    });
    container.appendChild(btn);
  });
}

async function fetchCatalog() {
  const grid = document.getElementById('modal-catalog-grid');
  grid.innerHTML = '<div style="color: #9ba5b9; padding: 20px;">Loading canonical LDraw thumbnails...</div>';
  
  try {
    const query = new URLSearchParams({
      category: currentCategory,
      search: currentSearch,
      page: catalogPage,
      limit: 48
    });
    const res = await fetch(`/api/catalog?${query}`);
    const data = await res.json();
    
    document.getElementById('modal-page-info').textContent = `Page ${data.page} of ${data.pages} (${data.total} parts)`;
    document.getElementById('modal-prev-page').disabled = (data.page <= 1);
    document.getElementById('modal-next-page').disabled = (data.page >= data.pages);

    grid.innerHTML = '';
    data.items.forEach(item => {
      const isSelected = selectedCatalogParts.has(item.id);
      const card = document.createElement('div');
      card.className = `catalog-item-card ${isSelected ? 'selected' : ''}`;
      card.innerHTML = `
        <img class="catalog-item-thumb" src="${item.thumbnail}" onerror="this.src='/static/thumbnails/3001.png'" alt="${item.id}">
        <div class="catalog-item-id">#${item.id}</div>
        <div class="catalog-item-name" title="${item.name}">${item.name}</div>
      `;
      card.addEventListener('click', () => {
        if (selectedCatalogParts.has(item.id)) {
          selectedCatalogParts.delete(item.id);
          card.classList.remove('selected');
        } else {
          selectedCatalogParts.add(item.id);
          card.classList.add('selected');
        }
        updateSelectedCounter();
      });
      grid.appendChild(card);
    });
    updateSelectedCounter();
  } catch (err) {
    grid.innerHTML = `<div style="color: #ef4444; padding: 20px;">Error loading catalog: ${err}</div>`;
  }
}

function updateSelectedCounter() {
  document.getElementById('modal-selected-count').textContent = `${selectedCatalogParts.size} parts selected`;
}

async function applySelectedPartsToBin() {
  if (activeModalBinId === null) return;
  const bin = allBins.find(b => b.id === activeModalBinId);
  if (bin) {
    bin.assigned_parts = Array.from(selectedCatalogParts);
    await updateBinProperty(bin.id, 'assigned_parts', bin.assigned_parts);
    loadBins();
    loadUnassignedBricks();
  }
  closePartPickerModal();
}

document.getElementById('modal-search')?.addEventListener('input', (e) => {
  currentSearch = e.target.value;
  catalogPage = 1;
  fetchCatalog();
});

document.getElementById('modal-prev-page')?.addEventListener('click', () => {
  if (catalogPage > 1) {
    catalogPage--;
    fetchCatalog();
  }
});

document.getElementById('modal-next-page')?.addEventListener('click', () => {
  catalogPage++;
  fetchCatalog();
});

// ----------------------------------------------------
// 8. Performance Metrics & Logs
// ----------------------------------------------------
function renderDistributionTable(dist) {
  const tbody = document.getElementById('distribution-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  allBins.forEach(bin => {
    const count = dist[bin.id] || 0;
    const tr = document.createElement('tr');
    const sideText = (bin.side || 'top').toUpperCase();
    tr.innerHTML = `
      <td><span class="badge" style="background: ${bin.color}22; color: ${bin.color}">Bin ${bin.id}</span></td>
      <td>${bin.name} (Spot ${bin.spot} ${sideText}, Tier ${bin.row})</td>
      <td><strong>${count}</strong></td>
      <td>${bin.delay_ms} ms delay</td>
      <td>${bin.assigned_parts ? bin.assigned_parts.length : 0} parts mapped</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderRecentDetectionsTable(detections) {
  const tbody = document.getElementById('recent-detections-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  detections.slice(0, 15).forEach(d => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${d.time_str}</td>
      <td>
        <img src="${d.thumbnail}" style="width: 22px; height: 22px; vertical-align: middle; margin-right: 6px;">
        #${d.part_id}
      </td>
      <td>${d.name}</td>
      <td><span class="badge ${d.bin_id === -1 ? 'btn-danger' : 'btn-success'}">Bin ${d.bin_id}</span></td>
      <td>${d.confidence_pct}%</td>
      <td>${d.delay_ms || 0} ms</td>
    `;
    tbody.appendChild(tr);
  });
}

async function pollLogs() {
  if (currentTab !== 'logs') return;
  try {
    const level = document.getElementById('log-filter-level')?.value || 'ALL';
    const res = await fetch(`/api/logs?level=${level}`);
    const data = await res.json();
    const terminal = document.getElementById('log-terminal');
    if (!terminal) return;

    terminal.innerHTML = '';
    data.logs.forEach(entry => {
      const line = document.createElement('div');
      line.className = `log-entry ${entry.level}`;
      line.textContent = `[${entry.time_str}] [${entry.level}] ${entry.logger}: ${entry.message}`;
      terminal.appendChild(line);
    });

    if (autoScrollLogs) {
      terminal.scrollTop = terminal.scrollHeight;
    }
  } catch (err) {
    console.error('Error fetching logs:', err);
  }
}

// ----------------------------------------------------
// 9. Settings Form & Shutdown
// ----------------------------------------------------
async function loadSettings() {
  try {
    const res = await fetch('/api/settings');
    const data = await res.json();
    const s = data.settings;

    const camSelect = document.getElementById('setting-camera');
    camSelect.innerHTML = '';
    data.available_cameras.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c.id;
      opt.textContent = c.name;
      if (String(c.id) === String(s.camera_source)) opt.selected = true;
      camSelect.appendChild(opt);
    });

    const serialSelect = document.getElementById('setting-serial-port');
    serialSelect.innerHTML = '';
    data.serial_ports.forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.port;
      opt.textContent = p.description;
      if (p.port === s.arduino_port) opt.selected = true;
      serialSelect.appendChild(opt);
    });

    document.getElementById('setting-thresh-mode').value = s.threshold_mode || 'adaptive';
    document.getElementById('setting-thresh-val').value = s.threshold_val || 120;
    document.getElementById('setting-thresh-val-num').textContent = s.threshold_val || 120;

    document.getElementById('setting-min-area').value = s.min_contour_area || 800;
    document.getElementById('setting-snapshot-count').value = s.snapshot_count || 5;

    document.getElementById('setting-baud').value = s.arduino_baud || 115200;
    document.getElementById('setting-gate-pulse').value = s.gate_pulse_ms || 350;

    document.getElementById('setting-model-path').value = s.model_path || '';
    document.getElementById('setting-labels-path').value = s.labels_path || '';
    document.getElementById('setting-conf-thresh').value = s.confidence_threshold || 0.35;
  } catch (err) {
    console.error('Error loading settings:', err);
  }
}

function initSettingsForm() {
  document.getElementById('setting-thresh-val')?.addEventListener('input', (e) => {
    document.getElementById('setting-thresh-val-num').textContent = e.target.value;
  });

  document.getElementById('settings-form')?.addEventListener('submit', async (e) => {
    e.preventDefault();
    const payload = {
      camera_source: parseInt(document.getElementById('setting-camera').value),
      threshold_mode: document.getElementById('setting-thresh-mode').value,
      threshold_val: parseInt(document.getElementById('setting-thresh-val').value),
      min_contour_area: parseInt(document.getElementById('setting-min-area').value),
      snapshot_count: parseInt(document.getElementById('setting-snapshot-count').value),
      arduino_port: document.getElementById('setting-serial-port').value,
      arduino_baud: parseInt(document.getElementById('setting-baud').value),
      gate_pulse_ms: parseInt(document.getElementById('setting-gate-pulse').value),
      model_path: document.getElementById('setting-model-path').value,
      labels_path: document.getElementById('setting-labels-path').value,
      confidence_threshold: parseFloat(document.getElementById('setting-conf-thresh').value)
    };

    try {
      const res = await fetch('/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (res.ok) {
        alert('Settings saved and applied successfully!');
      }
    } catch (err) {
      alert(`Error saving settings: ${err}`);
    }
  });
}

async function shutdownServer() {
  if (!confirm('Are you sure you want to SHUT DOWN the LEGO Sorter server? This will stop camera tracking, close serial connections, and unbind the network port.')) {
    return;
  }

  if (statusPollTimer) clearInterval(statusPollTimer);
  if (logsPollTimer) clearInterval(logsPollTimer);

  try {
    await fetch('/api/control/shutdown', { method: 'POST' });
  } catch (err) {
    // Server may terminate immediately before returning response
    console.warn('Shutdown response interrupted:', err);
  }

  document.body.innerHTML = `
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; background: #12151b; color: #fff; font-family: sans-serif; text-align: center; padding: 20px;">
      <h1 style="color: #ef4444; margin-bottom: 12px;">LEGO Sorter Server Stopped</h1>
      <p style="color: #9ba5b9; max-width: 500px; line-height: 1.5; margin-bottom: 20px;">
        The vision pipeline, camera capture, and Arduino serial connection have been cleanly closed. The network port has been released.
      </p>
      <p style="font-size: 0.85rem; color: #64748b;">You may now close this window. To restart, run <code>./start_sorter.sh</code>.</p>
    </div>
  `;

  // Attempt to close the standalone app window
  setTimeout(() => {
    try { window.close(); } catch (e) {}
  }, 300);
}
