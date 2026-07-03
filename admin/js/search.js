let activeDb = null;
let hasAllAccess = false;
let dbAccess = {};

const search = {
  getQuery: function () {
    return document.getElementById("searchQuery").value;
  },

  sendQuery: async function (query) {
    if (!activeDb) {
        alert("Please select a database to search in");
        return [];
    }

    const isImageSearch = document.getElementById("searchTypeImageToggle")?.checked;
    const searchType = isImageSearch ? "image" : "text";
    const dbName = activeDb === "*" ? "*" : activeDb;

    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: 'include',
        body: JSON.stringify({ query, dbName, searchType }),
      });

      if (response.redirected && response.url.includes('/login')) {
        window.location.href = '/login';
        return [];
      }

      if (!response.ok) throw new Error("Network response was not ok");

      const results = await response.json();
      return results;
    } catch (error) {
      console.error("Error:", error);
      return [];
    }
  },

  searchInText: function (searchText, previewElement) {
    const cardText = previewElement.innerHTML;
    const searchRegex = new RegExp(`(${searchText.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
    const highlightedText = cardText.replace(searchRegex, '<span class="highlight">$1</span>');
    previewElement.innerHTML = highlightedText;
  },

  displayResults: function (results, query) {
    const resultsContainer = document.getElementById("results");
    resultsContainer.innerHTML = "";
    if (results.length > 0) {
      const groupedResults = {};
      results.forEach(result => {
          const docId = result.document.id;
          if (!groupedResults[docId]) {
              groupedResults[docId] = {
                  document: result.document,
                  contexts: [],
                  maxScore: result.similarity
              };
          }

        let textContext = result.context;
        if (!textContext || textContext.trim() === "") {
            textContext = (result.document.meta && result.document.meta.content) || "No preview available.";
        }

        groupedResults[docId].contexts.push(textContext);
          if (result.similarity > groupedResults[docId].maxScore) {
              groupedResults[docId].maxScore = result.similarity;
          }
      });

      Object.values(groupedResults).forEach((group) => {
        const similarityScore = group.maxScore ? group.maxScore.toFixed(2) : "N/A";
        const color = similarityScore > 0.4 ? "green" : "red";

        const cardId = `result-card-${Date.now()}-${Math.floor(Math.random() * 1000)}`;

        const filePathOrHref = group.document.meta.filePath || group.document.meta.href || "";
        const isImage = filePathOrHref.match(/\.(jpeg|jpg|png|gif|webp)$/i) != null;
        const resultDb = group.document._db || document.getElementById('activeSearchDb').innerText;

        let contextsHtml = "";
        if (isImage) {
            const imgSrc = `/api/file?path=${encodeURIComponent(filePathOrHref)}`;
            contextsHtml += `
               <div class="text-center mb-3 mt-2">
                   <img src="${imgSrc}" class="img-fluid rounded shadow-sm" style="max-height: 300px; object-fit: contain; cursor: pointer;" onclick="window.open(this.src, '_blank')" alt="Image preview">
               </div>
            `;
            group.contexts = group.contexts.filter(c => c !== "No preview available.");
        }

        group.contexts.forEach(context => {
            contextsHtml += `<p class="card-text context-preview">${context}</p>`;
        });

        const card = `
              <div class="col-md-12 mb-3">
                  <div class="card" id="${cardId}">
                      <div class="card-header">
                          <div class="d-flex justify-content-between align-items-center w-100">
                              <div class="card-title">${(group.document.meta.href || group.document.meta.filePath || '').split("/").at(-1)}</div>
                              ${group.document._db ? `<span class="badge bg-secondary" style="font-size: 0.75rem; flex-shrink: 0;">${group.document._db}</span>` : ''}
                          </div>
                      </div>
                      <div class="card-body">
                          ${contextsHtml}
                          <div class="mt-3 d-flex justify-content-between align-items-center">
                              <div>
                                  <a target="_blank" href="/api/file?path=${encodeURIComponent(group.document.meta.filePath || group.document.meta.href)}" class="btn btn-custom-grey" style="min-width: 140px;">View Document</a>
                                  ${dbAccess[resultDb] === 'readWrite' ? `<button class="btn btn-danger ms-2" style="min-width: 140px;" onclick="search.deleteDocument('${resultDb}', '${group.document.id}')">Delete Document</button>` : ''}
                              </div>
                              <div class="similarity-score" ${color !== 'green' ? `style="color: #f87171; border-color: rgba(239, 68, 68, 0.25); background-color: rgba(239, 68, 68, 0.1);"` : ""}>Score: ${similarityScore}</div>
                          </div>
                      </div>
                  </div>
              </div>`;
        resultsContainer.innerHTML += card;

        document.getElementById(cardId).querySelectorAll(".context-preview").forEach((previewPara) => {
            search.searchInText(query, previewPara);
        });
      });
    } else {
      resultsContainer.innerHTML = "<p>No results found</p>";
    }
  },

  handleSubmit: async function (event) {
    if (event) event.preventDefault();
    const query = this.getQuery();
    if (query) {
      document.getElementById("results").innerHTML = "<p>Searching...</p>";
      const results = await this.sendQuery(query);
      this.displayResults(results, query);
    } else {
      alert("Please enter a search query");
    }
  },

  fetchDatabases: async function () {
      try {
          const res = await fetch('/api/search/databases', { credentials: 'include' });
          if (res.redirected) return window.location.href = '/login';
          if (!res.ok) throw new Error("Failed to fetch DBs");
          const dbs = await res.json();
          hasAllAccess = dbs.length > 1;
          dbAccess = {};
          dbs.forEach(db => { dbAccess[db.name] = db.access; });
          this.renderSidebar(dbs);
      } catch (err) {
          console.error(err);
      }
  },

  renderSidebar: function (dbs) {
      const tree = document.getElementById("dbTree");
      tree.innerHTML = "";

      if (hasAllAccess && dbs.length > 0) {
          const isAll = activeDb === "*";
          const allLi = document.createElement("li");
          allLi.innerHTML = `
            <div class="tree-node ${isAll ? 'active' : ''}" onclick="search.setActiveDb('*')">
                <div>
                    <i class="fas fa-globe"></i>
                    <span class="db-name">All Databases</span>
                    <small class="text-muted ml-2 db-count">(${dbs.reduce((s, d) => s + d.documentCount, 0)})</small>
                </div>
            </div>
          `;
          tree.appendChild(allLi);
      }

      dbs.forEach(db => {
          const li = document.createElement("li");
          const isActive = db.name === activeDb;
          const canManage = db.access === 'readWrite';

          li.innerHTML = `
            <div class="tree-node ${isActive ? 'active' : ''}" onclick="search.setActiveDb('${db.name}')">
                <div>
                    <i class="fas fa-database"></i>
                    <span class="db-name">${db.name}</span>
                    <small class="text-muted ml-2 db-count">(${db.documentCount})</small>
                </div>
                ${canManage ? `
                <div class="actions">
                    <button class="btn-icon" onclick="event.stopPropagation(); search.openUploadModal('${db.name}')" title="Upload Document">
                        <i class="fas fa-upload"></i>
                    </button>
                    <button class="btn-icon" onclick="event.stopPropagation(); search.manageAccess('${db.name}')" title="Manage Access" style="color:#60a5fa"><i class="fas fa-users"></i></button>
                    <button class="btn-icon delete" onclick="event.stopPropagation(); search.deleteDatabase('${db.name}')" title="Delete Database">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>` : ''}
            </div>
          `;

          tree.appendChild(li);
      });

      if (!activeDb) {
          if (hasAllAccess && dbs.length > 0) {
              this.setActiveDb("*");
          } else if (dbs.length > 0) {
              this.setActiveDb(dbs[0].name);
          }
      }
  },

  setActiveDb: function (name) {
      activeDb = name;
      document.getElementById("activeSearchDb").innerText = name === "*" ? "All Databases" : name;
      this.fetchDatabases();
  },

  createDatabase: async function () {
      const name = document.getElementById("newDbName").value.trim();
      if (!name) return alert("Enter a database name");

      try {
          const res = await fetch('/api/databases', {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              credentials: 'include',
              body: JSON.stringify({ name })
          });
          if (res.redirected) return window.location.href = '/login';
          if (res.ok) {
              bootstrap.Modal.getOrCreateInstance(document.getElementById('createDbModal')).hide();
              document.getElementById("newDbName").value = '';
              this.setActiveDb(name);
          } else {
              const data = await res.json();
              alert(data.error);
          }
      } catch (err) {
          console.error(err);
      }
  },

  deleteDatabase: async function (name) {
      if (!confirm(`Are you sure you want to delete database '${name}'?`)) return;
      try {
          const res = await fetch(`/api/databases/${name}`, {
              method: "DELETE",
              credentials: 'include'
          });
          if (res.redirected) return window.location.href = '/login';
          if (res.ok) {
              if (activeDb === name) activeDb = null;
              this.fetchDatabases();
          }
      } catch (err) {
          console.error(err);
      }
  },

  openUploadModal: function (dbName) {
      dbName = dbName || activeDb;
      if (!dbName) return alert("Select a database first");
      document.getElementById("uploadTargetDb").innerText = dbName;
      document.getElementById("uploadDocModal").dataset.targetDb = dbName;
      bootstrap.Modal.getOrCreateInstance(document.getElementById('uploadDocModal')).show();
  },

  openTextModal: function (dbName) {
      dbName = dbName || activeDb;
      if (!dbName) return alert("Select a database first");
      document.getElementById("textTargetDb").innerText = dbName;
      document.getElementById("pasteTextModal").dataset.targetDb = dbName;
      bootstrap.Modal.getOrCreateInstance(document.getElementById('pasteTextModal')).show();
  },

  uploadDocument: async function () {
      const dbName = document.getElementById("uploadDocModal").dataset.targetDb;
      const fileInput = document.getElementById("docFile");

      let filesToUpload = window.droppedFiles || Array.from(fileInput.files);
      if (!filesToUpload.length) return alert("Select at least one file first");

      const btn = document.getElementById("btnUploadDoc");
      btn.disabled = true;
      btn.innerText = `Uploading ${filesToUpload.length} file(s)...`;

      const progressContainer = document.getElementById("uploadProgressContainer");
      const progressBar = document.getElementById("uploadProgressBar");

      progressContainer.style.display = "block";
      progressBar.style.width = "0%";
      progressBar.setAttribute("aria-valuenow", 0);
      progressBar.innerText = "0%";
      progressBar.classList.add("progress-bar-animated");

      const errors = [];
      for (const file of filesToUpload) {
          try {
              btn.innerText = `[${errors.length+1}/${filesToUpload.length}] ${file.name}...`;
              const base64 = await new Promise((resolve, reject) => {
                  const r = new FileReader();
                  r.onload = () => resolve(r.result.split(',')[1]);
                  r.onerror = reject;
                  r.readAsDataURL(file);
              });
              const res = await fetch(`/api/databases/${dbName}/documents`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  credentials: 'include',
                  body: JSON.stringify({ data: base64, filename: file.name })
              });
              if (res.redirected) return window.location.href = '/login';
              if (!res.ok) {
                  const data = await res.json().catch(() => ({}));
                  errors.push(`${file.name}: ${data.error || res.statusText}`);
              }
          } catch(e) {
              errors.push(`${file.name}: ${e.message}`);
          }
      }

      progressContainer.style.display = "none";
      btn.disabled = false;
      btn.innerText = "Upload";

      if (errors.length === 0) {
          bootstrap.Modal.getOrCreateInstance(document.getElementById('uploadDocModal')).hide();
          fileInput.value = "";
          this.fetchDatabases();
      } else {
          alert(errors.join('\n'));
      }
  },

  uploadTextDocument: async function () {
      const dbName = document.getElementById("pasteTextModal").dataset.targetDb;
      const title = document.getElementById("docTitle").value.trim();
      const text = document.getElementById("docText").value.trim();

      if (!title || !text) return alert("Title and Content are required");

      const btn = document.getElementById("btnPasteText");
      btn.disabled = true;
      btn.innerText = "Saving...";

      try {
          const res = await fetch('/api/databases/text', {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              credentials: 'include',
              body: JSON.stringify({ dbName, title, text })
          });
          if (res.redirected) return window.location.href = '/login';

          if (res.ok) {
              bootstrap.Modal.getOrCreateInstance(document.getElementById('pasteTextModal')).hide();
              document.getElementById("docTitle").value = "";
              document.getElementById("docText").value = "";
              this.fetchDatabases();
          } else {
              const data = await res.json();
              alert(data.error);
          }
      } catch (err) {
          console.error(err);
      } finally {
          btn.disabled = false;
          btn.innerText = "Save";
      }
  },

  deleteDocument: async function (dbName, docId) {
      if (!confirm("Delete this document?")) return;
      try {
          const res = await fetch(`/api/databases/${dbName}/documents/${docId}`, {
              method: "DELETE",
              credentials: 'include'
          });
          if (res.redirected) return window.location.href = '/login';
          if (res.ok) {
              this.fetchDatabases();
              const query = this.getQuery();
              if (query) this.handleSubmit();
          } else {
               const data = await res.json();
               alert(data.error);
          }
      } catch (err) {
          console.error(err);
      }
  },

  manageAccess: async function (dbName) {
      try {
          const res = await fetch(`/api/search/databases/${dbName}/users`, { credentials: 'include' });
          if (res.redirected) return window.location.href = '/login';
          if (!res.ok) { const d = await res.json(); alert(d.error); return; }
          const users = await res.json();
          const list = document.getElementById('manageAccessUserList');
          const title = document.getElementById('manageAccessTitle');
          title.textContent = 'Access for: ' + dbName;
          list.innerHTML = users.map(u => `
            <div style="display:flex;align-items:center;justify-content:space-between;padding:8px 0;border-bottom:1px solid #32363b">
              <span>${u.user}</span>
              <select class="manage-access-select" data-db="${dbName}" data-user="${u.user}" style="background:#16181b;color:#e9ecef;border:1px solid #32363b;border-radius:4px;padding:4px 8px">
                <option value="read" ${u.access === 'read' ? 'selected' : ''}>Read</option>
                <option value="readWrite" ${u.access === 'readWrite' ? 'selected' : ''}>Read/Write</option>
                <option value="none">Remove</option>
              </select>
            </div>
          `).join('');
          if (!users.length) list.innerHTML = '<div style="color:#8b949e;padding:12px 0">No other users have access to this database</div>';
          document.getElementById('manageAccessModal').dataset.db = dbName;
          bootstrap.Modal.getOrCreateInstance(document.getElementById('manageAccessModal')).show();
      } catch (err) { console.error(err); }
  },

  saveManageAccess: async function () {
      const dbName = document.getElementById('manageAccessModal').dataset.db;
      const selects = document.querySelectorAll('.manage-access-select');
      for (const sel of selects) {
          const targetUser = sel.dataset.user;
          const access = sel.value;
          try {
              const res = await fetch(`/api/search/databases/${dbName}/users/${targetUser}`, {
                  method: 'PUT',
                  headers: { 'Content-Type': 'application/json' },
                  credentials: 'include',
                  body: JSON.stringify({ access })
              });
              if (res.redirected) return window.location.href = '/login';
              if (!res.ok) { const d = await res.json(); alert(d.error); }
          } catch (err) { console.error(err); }
      }
      bootstrap.Modal.getOrCreateInstance(document.getElementById('manageAccessModal')).hide();
  }
};

document.addEventListener('DOMContentLoaded', () => {
    search.fetchDatabases();

    document.getElementById("btnCreateDb")?.addEventListener("click", () => search.createDatabase());
    document.getElementById("btnUploadDoc")?.addEventListener("click", () => search.uploadDocument());

    const btnPasteText = document.getElementById("btnPasteText");
    if (btnPasteText) {
        btnPasteText.addEventListener("click", () => search.uploadTextDocument());
    }

    const btnImageSearch = document.getElementById("btnImageSearch");
    const imageSearchInput = document.getElementById("imageSearchInput");

    if (btnImageSearch && imageSearchInput) {
        btnImageSearch.addEventListener("click", () => imageSearchInput.click());

        imageSearchInput.addEventListener("change", async (e) => {
            if (e.target.files && e.target.files.length > 0) {
                const file = e.target.files[0];
                document.getElementById("results").innerHTML = "<p>Searching by image...</p>";

                const formData = new FormData();
                formData.append("image", file);
                formData.append("dbName", activeDb === "*" ? "*" : activeDb || "Main");

                try {
                    const response = await fetch('/api/search/image', {
                        method: "POST",
                        body: formData,
                        credentials: 'include'
                    });
                    if (response.redirected && response.url.includes('/login')) {
                        window.location.href = '/login';
                        return;
                    }
                    if (!response.ok) {
                        const errData = await response.json();
                        throw new Error(errData.error || "Image search failed");
                    }
                    const results = await response.json();
                    search.displayResults(results, `Image: ${file.name}`);
                } catch (error) {
                    console.error("Error:", error);
                    document.getElementById("results").innerHTML = `<p class="text-danger">Error: ${error.message}</p>`;
                }
                imageSearchInput.value = "";
            }
        });
    }

    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("docFile");

    if (dropZone && fileInput) {
        dropZone.addEventListener("click", () => fileInput.click());

        fileInput.addEventListener("change", (e) => {
            window.droppedFiles = Array.from(e.target.files);
            if (window.droppedFiles.length > 0) {
                dropZone.querySelector("p.text-muted").innerText = `${window.droppedFiles.length} file(s) selected`;
                dropZone.style.borderColor = "#60a5fa";
            }
        });

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
            document.body.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.add("dragover"), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, () => dropZone.classList.remove("dragover"), false);
        });

        dropZone.addEventListener('drop', (e) => {
            window.droppedFiles = Array.from(e.dataTransfer.files);
            if (window.droppedFiles.length > 0) {
                dropZone.querySelector("p.text-muted").innerText = `${window.droppedFiles.length} file(s) selected`;
                dropZone.style.borderColor = "#60a5fa";
            }
        }, false);

        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
    }

    document.getElementById('uploadDocModal')?.addEventListener('hidden.bs.modal', function () {
        window.droppedFiles = null;
        if(fileInput) fileInput.value = '';
        if(dropZone) {
            dropZone.querySelector("p.text-muted").innerText = "or click to browse";
            dropZone.style.borderColor = "";
        }
        document.getElementById("uploadProgressContainer").style.display = "none";
        document.getElementById("btnUploadDoc").disabled = false;
        document.getElementById("btnUploadDoc").innerText = "Upload";
    });
});
