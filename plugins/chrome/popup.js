if (typeof browser === "undefined") {
  window.browser = {
    runtime: {
      sendMessage: (msg) => new Promise((resolve, reject) => {
        chrome.runtime.sendMessage(msg, (response) => {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
          } else {
            resolve(response);
          }
        });
      })
    },
    storage: {
      local: {
        get: (keys) => new Promise((resolve) => {
          chrome.storage.local.get(keys, (items) => resolve(items));
        }),
        set: (items) => new Promise((resolve) => {
          chrome.storage.local.set(items, () => resolve());
        }),
        remove: (keys) => new Promise((resolve) => {
          chrome.storage.local.remove(keys, () => resolve());
        })
      }
    }
  };
}

class RecallPanel {
  constructor() {
    this.queryInput = document.getElementById("query");
    this.searchButton = document.getElementById("search");
    this.addSelectionButton = document.getElementById("add-selection");
    this.addPageSnapshotButton = document.getElementById("add-page-snapshot");
    this.addFileInput = document.getElementById("add-file");
    this.resultsList = document.getElementById("results");
    this.configBtn = document.getElementById("config-btn");
    this.configPanel = document.getElementById("config-panel");
    this.cfgServerUrl = document.getElementById("cfg-server-url");
    this.cfgUsername = document.getElementById("cfg-username");
    this.cfgPassword = document.getElementById("cfg-password");
    this.cfgDbName = document.getElementById("cfg-dbname");
    this.cfgRerank = document.getElementById("cfg-rerank");
    this.cfgTestBtn = document.getElementById("cfg-test-btn");
    this.cfgSaveBtn = document.getElementById("cfg-save-btn");
    this.cfgStatus = document.getElementById("cfg-status");
    this.cfgFetchDbs = document.getElementById("cfg-fetch-dbs");
    this.dbSelect = document.getElementById("db-select");
    this.imageToggle = document.getElementById("search-type-image");
    this.dbAccess = {};
    this.headerTitle = document.getElementById("header-title");

    this.config = {
      serverUrl: "http://localhost:3030",
      username: "",
      password: "",
      dbName: "",
      searchDb: "*",
      rerank: false,
      searchImages: false,
    };

    this._applyingConfig = false;
    this.bindEvents();
    this.loadConfig().then(() => {
      this._applyingConfig = false;
      this.checkPendingSearch();
      if (this.config.serverUrl && this.config.username) {
        this.fetchDatabases();
      }
    });
  }

  async loadConfig() {
    try {
      const stored = await browser.storage.local.get("hiveConfig");
      if (stored.hiveConfig) {
        this.config = { ...this.config, ...stored.hiveConfig };
      }
    } catch (e) {
      console.error("Failed to load config:", e);
    }
    this.applyConfigToUI();
  }

  async saveConfig() {
    this.readConfigFromUI();
    this.cfgSaveBtn.disabled = true;
    this.cfgStatus.className = "loading";
    this.cfgStatus.textContent = "Connecting...";

    try {
      const ensure = await this.ensureDatabase();
      if (!ensure.success && this.config.username) {
        this.cfgStatus.className = "error";
        this.cfgStatus.textContent = "Auto-provision failed: " + ensure.error + " — saved anyway";
      }

      await browser.storage.local.set({ hiveConfig: this.config });
      browser.runtime.sendMessage({ type: "CONFIG_UPDATED" });
      this.cfgStatus.className = "success";
      this.cfgStatus.textContent = "Settings saved" + (ensure.success ? " — database ready" : "");
      setTimeout(() => { if (this.cfgStatus.className === "success") this.cfgStatus.textContent = ""; }, 3000);

      if (ensure.success) {
        await this.fetchDatabases();
      }
    } catch (e) {
      this.cfgStatus.className = "error";
      this.cfgStatus.textContent = "Failed to save: " + e.message;
    } finally {
      this.cfgSaveBtn.disabled = false;
    }
  }

  readConfigFromUI() {
    const url = this.cfgServerUrl ? this.cfgServerUrl.value.trim() : "";
    this.config.serverUrl = url || this.config.serverUrl;
    this.config.username = this.cfgUsername ? this.cfgUsername.value.trim() : "";
    this.config.password = this.cfgPassword ? this.cfgPassword.value : "";
    this.config.dbName = (this.cfgDbName ? this.cfgDbName.value.trim() : "") || this.config.username || this.config.dbName;
    this.config.searchDb = this.config.searchDb || "*";
    this.config.rerank = this.cfgRerank ? this.cfgRerank.checked : false;
    this.config.searchImages = this.imageToggle ? this.imageToggle.checked : false;
  }

  applyConfigToUI() {
    this._applyingConfig = true;
    this.cfgServerUrl.value = this.config.serverUrl;
    this.cfgUsername.value = this.config.username;
    this.cfgPassword.value = this.config.password;
    this.cfgDbName.value = this.config.dbName;
    this.cfgRerank.checked = this.config.rerank;
    if (this.imageToggle) this.imageToggle.checked = !!this.config.searchImages;
    this.dbSelect.value = this.config.searchDb || "*";
    this._applyingConfig = false;
  }

  async bg(type, msg = {}) {
    return browser.runtime.sendMessage({ type, ...msg });
  }

  async checkPendingSearch() {
    const { pendingSearch } = await browser.storage.local.get("pendingSearch");
    if (pendingSearch) {
      this.queryInput.value = pendingSearch;
      this.performSearch();
      browser.storage.local.remove("pendingSearch");
    }
  }

  bindEvents() {
    this.searchButton.addEventListener("click", () => this.performSearch());
    this.queryInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter") this.performSearch();
    });
    this.addSelectionButton.addEventListener("click", () => this.addSelection());
    this.addPageSnapshotButton.addEventListener("click", () => this.addPageSnapshot());
    this.addFileInput.addEventListener("change", (e) => this.addFile(e));
    this.configBtn.addEventListener("click", () => this.toggleConfig());
    this.cfgSaveBtn.addEventListener("click", () => this.saveConfig());
    this.cfgTestBtn.addEventListener("click", () => this.testConnection());
    this.cfgFetchDbs.addEventListener("click", () => this.fetchDatabases());
    this.dbSelect.addEventListener("change", () => {
      if (this._applyingConfig) return;
      this.config.searchDb = this.dbSelect.value;
    });
    if (this.imageToggle) {
      this.imageToggle.addEventListener("change", () => {
        this.config.searchImages = this.imageToggle.checked;
        browser.storage.local.set({ hiveConfig: this.config });
      });
    }


  }

  toggleConfig() {
    const shown = this.configPanel.style.display !== "none";
    this.configPanel.style.display = shown ? "none" : "block";
    this.configBtn.style.opacity = shown ? "0.7" : "1";
    if (!shown) {
      this.applyConfigToUI();
      this.cfgStatus.textContent = "";
    }
  }

  async testConnection() {
    this.readConfigFromUI();
    this.cfgStatus.className = "loading";
    this.cfgStatus.textContent = "Testing connection...";
    this.cfgTestBtn.disabled = true;

    try {
      const result = await this.ensureDatabase();
      if (result.success) {
        this.cfgStatus.className = "success";
        this.cfgStatus.textContent = "Connected — database '" + this.config.dbName + "' ready";
        await this.fetchDatabases();
      } else {
        this.cfgStatus.className = "error";
        this.cfgStatus.textContent = "Connection failed: " + (result.error || "Unknown error");
      }
    } catch (e) {
      this.cfgStatus.className = "error";
      this.cfgStatus.textContent = "Connection failed: " + e.message;
    } finally {
      this.cfgTestBtn.disabled = false;
    }
  }

  async ensureDatabase() {
    try {
      const result = await browser.runtime.sendMessage({
        type: "ENSURE_DATABASE",
        serverUrl: this.config.serverUrl,
        username: this.config.username,
        password: this.config.password,
        dbName: this.config.dbName,
      });
      return result;
    } catch (e) {
      return { success: false, error: e.message };
    }
  }

  async fetchDatabases() {
    const res = await this.bg("FETCH_DATABASES");
    if (!res.success) {
      console.warn("fetchDatabases failed:", res.error);
      return;
    }

    let list = res.databases;
    if (list && !Array.isArray(list) && list.databases) {
      list = list.databases;
    }
    if (!Array.isArray(list)) {
      list = [];
    }

    this.dbAccess = {};
    const current = this.dbSelect.value;
    this.dbSelect.innerHTML = '<option value="*">All databases</option>';
    for (const db of list) {
      const name = typeof db === "string" ? db : db.name;
      const access = db.access || "";
      const count = db.documentCount ?? "";
      this.dbAccess[name] = access;
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name + (count !== "" ? " (" + count + ")" : "");
      this.dbSelect.appendChild(opt);
    }
    if ([...this.dbSelect.options].some(o => o.value === current)) {
      this.dbSelect.value = current;
    }
    this.config.searchDb = this.dbSelect.value;
  }

  async performSearch() {
    const query = this.queryInput.value.trim();
    if (!query) return;

    this.showLoading();

    const res = await this.bg("SEARCH", { query, dbName: this.dbSelect.value, topK: 10, searchType: (this.imageToggle && this.imageToggle.checked) ? "image" : "text" });
    if (res.success) {
      this.displayResults(res.results);
    } else {
      this.showError(res.error);
    }
  }

  showLoading() {
    this.resultsList.innerHTML = '<li class="loading">Searching...</li>';
  }

  showError(message) {
    this.resultsList.innerHTML = `<li class="error">Error: ${message}</li>`;
  }

  fileUrl(path) {
    if (!path) return "";
    if (/^https?:\/\//i.test(path)) return path;
    const base = (this.config.serverUrl || "").replace(/\/+$/, "");
    return base + "/api/file?path=" + encodeURIComponent(path);
  }

  openImage(url) {
    try {
      if (typeof browser !== "undefined" && browser.tabs && browser.tabs.create) {
        browser.tabs.create({ url });
        return;
      }
      if (typeof chrome !== "undefined" && chrome.tabs && chrome.tabs.create) {
        chrome.tabs.create({ url });
        return;
      }
    } catch (e) { /* fall through */ }
    window.open(url, "_blank");
  }

  displayResults(results) {
    this.resultsList.innerHTML = "";

    if (!results || results.length === 0) {
      this.resultsList.innerHTML = '<li class="no-results">No results found</li>';
      return;
    }

    results.forEach((result) => {
      const item = document.createElement("li");
      item.className = "result-item";

      const doc = result.document || {};
      const meta = doc.meta || {};
      const content = meta.content || "(no content)";
      const href = meta.href || meta.filePath || "";
      const fileName = href.split("/").filter(Boolean).pop() || meta.title || "(no title)";
      const docId = doc.id || "";
      const resultDb = doc._db || "";
      const canDelete = this.dbAccess[resultDb] === "readWrite";
      const similarity = result.similarity || 0;
      const previewLen = 300;
      const isLong = content.length > previewLen;
      const preview = content.substring(0, previewLen);
      const isImage = meta.type === "image" || /\.(jpe?g|png|gif|webp|svg)$/i.test(href);
      const imageUrl = (isImage && href) ? this.fileUrl(href) : "";

      item.innerHTML = `
        <div class="result-header">
          <span class="similarity-badge">${(similarity * 100).toFixed(1)}%</span>
          <span class="result-title">${this.esc(fileName.substring(0, 40))}</span>
        </div>
        <div class="result-meta">
          <span class="result-source">${this.esc(href.substring(0, 30))}</span>
          ${resultDb ? `<span class="result-db">[${this.esc(resultDb)}]</span>` : ''}
        </div>
        <div class="result-content">
          ${imageUrl ? `<div class="result-image"><img src="${this.esc(imageUrl)}" alt="${this.esc(fileName)}" loading="lazy"></div>` : ''}
          <span class="result-text preview">${this.esc(preview)}${isLong ? '...' : ''}</span>
          ${isLong ? `<span class="result-text full" style="display:none">${this.esc(content)}</span>` : ''}
          ${isLong ? '<button class="toggle-text-btn">Show more</button>' : ''}
        </div>
        <div class="result-actions">
          <button class="copy-btn">Copy</button>
          <button class="insert-btn">Insert</button>
          <button class="download-btn">Download</button>
          ${canDelete ? '<button class="delete-btn">Delete</button>' : ''}
        </div>
      `;

      const copyBtn = item.querySelector(".copy-btn");
      const insertBtn = item.querySelector(".insert-btn");
      const toggleBtn = item.querySelector(".toggle-text-btn");
      const downloadBtn = item.querySelector(".download-btn");
      const deleteBtn = canDelete ? item.querySelector(".delete-btn") : null;

      copyBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.copyText(content);
      });

      insertBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.insertText(content);
      });

      if (toggleBtn) {
        toggleBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          const previewEl = item.querySelector(".result-text.preview");
          const fullEl = item.querySelector(".result-text.full");
          const expanded = fullEl.style.display !== "none";
          fullEl.style.display = expanded ? "none" : "block";
          previewEl.style.display = expanded ? "inline" : "none";
          toggleBtn.textContent = expanded ? "Show more" : "Show less";
        });
      }

      downloadBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        this.downloadFile(href, content);
      });

      if (deleteBtn) {
        deleteBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.deleteDocument(docId, resultDb, item);
        });
      }

      if (imageUrl) {
        const imgEl = item.querySelector(".result-image img");
        if (imgEl) {
          imgEl.addEventListener("click", (e) => {
            e.stopPropagation();
            this.openImage(imageUrl);
          });
        }
      }

      this.resultsList.appendChild(item);
    });
  }

  esc(text) {
    const d = document.createElement("div");
    d.textContent = text;
    return d.innerHTML;
  }

  async copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
      this.feedback("Copied!");
    } catch (e) {
      try {
        await browser.runtime.sendMessage({ type: "EXEC_CODE", text: text });
        this.feedback("Copied!");
      } catch (e2) {
        console.error("Copy failed:", e2);
      }
    }
  }

  async insertText(text) {
    try {
      const result = await browser.runtime.sendMessage({ type: "INSERT_TEXT", text });
      if (result && result.inserted) {
        this.feedback("Inserted!");
      } else {
        this.copyText(text);
      }
    } catch (e) {
      console.error("Insert failed:", e);
      this.copyText(text);
    }
  }

  async downloadFile(filePath, contentFallback) {
    const res = await this.bg("DOWNLOAD_FILE", { filePath, content: contentFallback });
    if (res && res.success) {
      this.feedback("Downloaded!");
    } else {
      this.feedback("Download failed");
    }
  }

  async deleteDocument(id, dbName, item) {
    if (!confirm("Are you sure you want to delete this document?")) return;
    const res = await this.bg("DELETE_DOC", { id, dbName });
    if (res.success) {
      item.remove();
      this.feedback("Deleted!");
    } else {
      this.feedback("Delete failed" + (res.error ? ": " + res.error : ""));
    }
  }

  async addSelection() {
    try {
      const selectedText = await browser.runtime.sendMessage({ type: "GET_SELECTION" });
      if (selectedText && selectedText.trim()) {
        const res = await this.bg("ADD_TEXT", { text: selectedText, url: "", title: "" });
        if (res.success) {
          this.feedback("Selection added!");
        } else {
          this.feedback("Add failed: " + (res.error || "unknown error"));
        }
      }
    } catch (e) {
      console.error("Add selection failed:", e);
      this.feedback("Add failed");
    }
  }

  async addPageSnapshot() {
    const res = await this.bg("ADD_PAGE_SNAPSHOT");
    if (res && res.success) {
      this.feedback("Snapshot added!");
    } else {
      this.feedback("Snapshot failed: " + ((res && res.error) || "unknown error"));
    }
  }

  async addFile(event) {
    const input = event.target;
    const files = [...input.files];
    if (!files.length) return;

    let done = 0, failed = 0;
    this.feedback("Uploading 0/" + files.length + "...");

    for (const file of files) {
      try {
        const base64 = await new Promise((resolve, reject) => {
          const r = new FileReader();
          r.onload = () => resolve(r.result.split(",")[1]);
          r.onerror = reject;
          r.readAsDataURL(file);
        });

        const res = await this.bg("UPLOAD_FILE", { base64, filename: file.name });
        if (res.success) {
          done++;
        } else {
          failed++;
          console.error("Upload failed for", file.name, res.error);
        }
      } catch (e) {
        failed++;
        console.error("Upload failed for", file.name, e);
      }
      this.feedback("Uploading " + (done + failed) + "/" + files.length + "...");
    }

    this.feedback(done + " uploaded" + (failed ? ", " + failed + " failed" : ""));
    input.value = "";
  }

  feedback(text) {
    const header = document.getElementById("header");
    const el = this.headerTitle;
    const origEl = el.textContent;
    const origHeader = header.style.background;
    const origColor = el.style.color;
    el.textContent = text;
    el.style.color = "#fff";
    header.style.background = "linear-gradient(135deg, #28a745, #20c997)";
    setTimeout(() => {
      el.textContent = origEl;
      el.style.color = origColor;
      header.style.background = origHeader;
    }, 1500);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  try {
    new RecallPanel();
  } catch (e) {
    console.error("RecallPanel init error:", e);
    const el = document.getElementById("header-title");
    if (el) el.textContent = "Error: " + e.message;
  }
});
