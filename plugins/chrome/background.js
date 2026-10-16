let config = {
  serverUrl: "http://localhost:3030",
  username: "",
  password: "",
  dbName: "",
};
let sessionCookie = "";

async function loadConfig() {
  try {
    const stored = await browser.storage.local.get("hiveConfig");
    if (stored.hiveConfig) {
      config = { ...config, ...stored.hiveConfig };
    }
  } catch (e) {
    console.error("Failed to load config:", e);
  }
}

async function getApiUrl() {
  await ready;
  return config.serverUrl.replace(/\/+$/, "");
}

async function reconnect() {
  if (!config.username || !config.password) {
    throw new Error("No credentials configured");
  }
  const result = await ensureDatabase(
    config.serverUrl, config.username, config.password, config.dbName
  );
  return result;
}

async function adminFetch(path, options = {}) {
  const baseUrl = await getApiUrl();
  if (!baseUrl) throw new Error("Server URL not configured");
  const url = baseUrl + path;
  const headers = { ...options.headers };
  if (sessionCookie) {
    headers["Cookie"] = sessionCookie;
  }
  let res;
  try {
    res = await fetch(url, { ...options, headers });
  } catch (e) {
    throw new Error("Network error: " + e.message + " (" + url + ")");
  }

  if (res.status === 401 || res.status === 403) {
    if (config.username && config.password) {
      try {
        await reconnect();
      } catch (re) {
        throw new Error("Re-login failed: " + re.message);
      }
      headers["Cookie"] = sessionCookie;
      try {
        res = await fetch(url, { ...options, headers });
      } catch (e) {
        throw new Error("Network error on retry: " + e.message);
      }
      if (res.status === 401 || res.status === 403) {
        throw new Error("Not authenticated — check credentials in settings");
      }
      return res;
    }
    throw new Error("Not authenticated — set credentials in settings");
  }
  return res;
}

async function ensureDatabase(serverUrl, username, password, dbName) {
  const base = serverUrl.replace(/\/+$/, "");
  dbName = dbName || username || "default";
  let cookie = "";

  if (username && password) {
    const loginRes = await fetch(base + "/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
      },
      body: JSON.stringify({ username, password }),
    });

    if (!loginRes.ok) {
      const text = await loginRes.text().catch(() => "");
      throw new Error("Login failed (HTTP " + loginRes.status + "): " + text.slice(0, 200));
    }

    const ct = (loginRes.headers.get("content-type") || "").toLowerCase();
    if (!ct.includes("application/json")) {
      const text = await loginRes.text().catch(() => "");
      throw new Error("Login failed: " + (text.slice(0, 200) || "Invalid credentials"));
    }

    const json = await loginRes.json();
    if (!json.success) {
      throw new Error("Login failed: " + (json.error || "Invalid credentials"));
    }

    const setCookie = loginRes.headers.get("set-cookie");
    cookie = setCookie ? setCookie.split(";")[0] : "";
  }

  const dbRes = await fetch(base + "/api/databases", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(cookie ? { "Cookie": cookie } : {}),
    },
    body: JSON.stringify({ name: dbName }),
  });

  if (dbRes.status !== 200 && dbRes.status !== 201 && dbRes.status !== 409) {
    const text = await dbRes.text().catch(() => "");
    throw new Error("Database creation failed: " + (text || dbRes.statusText));
  }

  sessionCookie = cookie;
  config.serverUrl = serverUrl;
  config.dbName = dbName;
  config.username = username;
  config.password = password;

  console.log("Database ensured:", dbName);
  return { cookie, dbName };
}

async function addTextDocument(text, url, title, dbName) {
  const targetDb = dbName || config.dbName || config.username || "default";
  const docTitle = (title && title !== "undefined") ? title : "";
  console.log("addTextDocument: targetDb=" + targetDb + " serverUrl=" + config.serverUrl);
  const res = await adminFetch("/api/databases/text", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dbName: targetDb, title: docTitle, text }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || "Failed to add document");
  }
  return res.json();
}

const ready = loadConfig().catch(e => console.error("Config load error:", e)).then(() => {
  browser.contextMenus.removeAll().then(() => {
    browser.contextMenus.create({
      id: "add-selection-to-hive",
      title: "Add to DocuSearch",
      contexts: ["selection"],
      icons: { "16": "icon-16.png", "32": "icon-32.png" }
    });
    browser.contextMenus.create({
      id: "add-page-to-hive",
      title: "Snapshot page to DocuSearch",
      contexts: ["page"],
      icons: { "16": "icon-16.png", "32": "icon-32.png" }
    });
  });
});

browser.contextMenus.onClicked.addListener((info, tab) => {
  loadConfig().then(() => {
    if (info.menuItemId === "add-selection-to-hive") {
      addTextDocument(info.selectionText, tab.url, tab.title)
        .then(d => console.log("Added:", d))
        .catch(e => console.error("Failed to add:", e));
    } else if (info.menuItemId === "add-page-to-hive") {
      browser.tabs.executeScript(tab.id, { code: "document.body.innerText" })
        .then(results => addTextDocument(results[0], tab.url, tab.title))
        .then(d => console.log("Snapshot added:", d))
        .catch(e => console.error("Failed to snapshot:", e));
    }
  });
});

browser.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "ENSURE_DATABASE") {
    ensureDatabase(message.serverUrl, message.username, message.password, message.dbName)
      .then(result => sendResponse({ success: true, ...result }))
      .catch(e => {
        console.error("ensureDatabase error:", e.message);
        sendResponse({ success: false, error: e.message });
      });
    return true;
  }

  if (message.type === "CONFIG_UPDATED") {
    loadConfig().then(() => {
      console.log("Config reloaded:", config.serverUrl, config.dbName);
      sendResponse({ success: true });
    });
    return true;
  }

  if (message.type === "SEARCH") {
    const searchDb = message.dbName || config.dbName;
    adminFetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: message.query, dbName: searchDb, topK: message.topK || 10, searchType: message.searchType || "text" }),
    })
    .then(r => r.json())
    .then(results => sendResponse({ success: true, results }))
    .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (message.type === "FETCH_DATABASES") {
    adminFetch("/api/search/databases")
    .then(r => r.json())
    .then(list => sendResponse({ success: true, databases: list }))
    .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (message.type === "ADD_TEXT") {
    (async () => {
      await loadConfig();
      let title = message.title;
      if (!title || title === "undefined") {
        const tabs = await browser.tabs.query({ active: true, currentWindow: true }).catch(() => []);
        title = (tabs[0]?.title) || "";
      }
      const d = await addTextDocument(message.text, message.url, title, message.dbName);
      return d;
    })()
      .then(d => sendResponse({ success: true, data: d }))
      .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (message.type === "DELETE_DOC") {
    const deleteDb = message.dbName || config.dbName;
    adminFetch("/api/databases/" + encodeURIComponent(deleteDb) + "/documents/" + encodeURIComponent(message.id), {
      method: "DELETE",
    })
    .then(r => {
      if (!r.ok) return r.json().then(e => { throw new Error(e.error); });
      sendResponse({ success: true });
    })
    .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (message.type === "UPLOAD_FILE") {
    adminFetch("/api/databases/" + encodeURIComponent(config.dbName) + "/documents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ data: message.base64, filename: message.filename, title: message.filename }),
    })
    .then(r => {
      if (!r.ok) return r.json().then(e => { throw new Error(e.error); });
      return r.json();
    })
    .then(d => sendResponse({ success: true, data: d }))
    .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (message.type === "DOWNLOAD_FILE") {
    const fp = message.filePath;

    async function tryServerFile() {
      const res = await adminFetch("/api/file?path=" + encodeURIComponent(fp));
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      await browser.downloads.download({ url, filename: fp.split("/").filter(Boolean).pop() || "download" });
      URL.revokeObjectURL(url);
    }

    async function tryDirectUrl() {
      const res = await fetch(fp);
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const ext = fp.match(/\.(\w+)(?:\?|#|$)/)?.[1] || "txt";
      await browser.downloads.download({ url, filename: "download." + ext });
      URL.revokeObjectURL(url);
    }

    async function tryTextFallback() {
      const blob = new Blob([message.content || "No content"], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      await browser.downloads.download({ url, filename: "document.txt" });
      URL.revokeObjectURL(url);
    }

    (async () => {
      if (fp.startsWith("http://") || fp.startsWith("https://")) {
        try {
          if (!fp.includes("//localhost") && !fp.includes("//10.") && !fp.includes("//127.")) {
            const test = await adminFetch("/api/file?path=" + encodeURIComponent(fp));
            if (test.ok) {
              const blob = await test.blob();
              const url = URL.createObjectURL(blob);
              await browser.downloads.download({ url, filename: fp.split("/").filter(Boolean).pop() || "download" });
              URL.revokeObjectURL(url);
              return sendResponse({ success: true });
            }
          }
          await tryDirectUrl();
        } catch {
          await tryTextFallback();
        }
      } else if (fp) {
        try {
          await tryServerFile();
        } catch {
          await tryTextFallback();
        }
      } else {
        await tryTextFallback();
      }
      sendResponse({ success: true });
    })().catch(e => {
      console.error("Download failed:", e.message);
      sendResponse({ success: false, error: e.message });
    });
    return true;
  }

  if (message.type === "ADD_TO_HIVE") {
    const p = message.payload;
    addTextDocument(p.text, p.url, p.title)
      .then(() => sendResponse({ success: true }))
      .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (message.type === "ADD_PAGE_SNAPSHOT") {
    (async () => {
      await loadConfig();
      const tabs = await browser.tabs.query({ active: true, currentWindow: true });
      const tab = tabs[0];
      if (!tab) throw new Error("No active tab found");
      const results = await browser.tabs.executeScript(tab.id, { code: "document.body.innerText" });
      await addTextDocument(results[0], tab.url, tab.title);
    })()
      .then(() => sendResponse({ success: true }))
      .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (message.type === "INSERT_TEXT") {
    browser.tabs.query({ active: true, currentWindow: true }).then(tabs => {
      const tab = tabs[0];
      const text = JSON.stringify(message.text);
      browser.tabs.executeScript(tab.id, {
        code: `(() => {
          const text = ${text};
          const el = document.activeElement;
          if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) {
            const s = el.selectionStart, e = el.selectionEnd;
            el.value = el.value.slice(0, s) + text + el.value.slice(e);
            el.setSelectionRange(s + text.length, s + text.length);
            el.dispatchEvent(new Event("input", { bubbles: true }));
            el.dispatchEvent(new Event("change", { bubbles: true }));
            return { inserted: true };
          }
          if (el && el.contentEditable === "true") {
            el.focus();
            document.execCommand("insertText", false, text);
            return { inserted: true };
          }
          return { inserted: false };
        })()`
      }).then(([result]) => sendResponse(result))
        .catch(() => sendResponse({ inserted: false, error: true }));
    });
    return true;
  }

  if (message.type === "GET_SELECTION") {
    browser.tabs.query({ active: true, currentWindow: true }).then(tabs => {
      const tab = tabs[0];
      browser.tabs.executeScript(tab.id, { code: "window.getSelection().toString()" })
        .then(([text]) => sendResponse(text))
        .catch(() => sendResponse(""));
    });
    return true;
  }

  if (message.type === "EXEC_CODE") {
    browser.tabs.query({ active: true, currentWindow: true }).then(tabs => {
      const tab = tabs[0];
      browser.tabs.executeScript(tab.id, { code: message.code })
        .then(([result]) => sendResponse(result))
        .catch(e => sendResponse({ error: e.message }));
    });
    return true;
  }
});

browser.omnibox.setDefaultSuggestion({ description: "Search DocuSearch database" });

browser.omnibox.onInputChanged.addListener((text, addSuggestions) => {
  if (!text.trim()) { addSuggestions([]); return; }

  adminFetch("/api/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query: text, dbName: config.dbName, topK: 5 }),
  })
  .then(r => r.json())
  .then(results => {
    const suggestions = (results || []).map(r => ({
      content: text,
      description: `${r.document?.meta?.title || "Untitled"} <dim>${((r.similarity || 0) * 100).toFixed(0)}%</dim>`
    }));
    addSuggestions(suggestions);
  })
  .catch(() => addSuggestions([]));
});

browser.omnibox.onInputEntered.addListener((text) => {
  browser.storage.local.set({ pendingSearch: text });
});
