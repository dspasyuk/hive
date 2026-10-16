/* webextension-polyfill - v0.12.0 - Tue May 14 2024 18:01:29 */
/* -*- Mode: indent-tabs-mode: nil; js-indent-level: 2 -*- */
/* vim: set sts=2 sw=2 et tw=80: */
/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/. */
"use strict";

if (!(globalThis.chrome && globalThis.chrome.runtime && globalThis.chrome.runtime.id)) {
  throw new Error("This script should only be loaded in a browser extension.");
}
if (!(globalThis.browser && globalThis.browser.runtime && globalThis.browser.runtime.id)) {
  const CHROME_SEND_MESSAGE_CALLBACK_NO_RESPONSE_MESSAGE = "The message port closed before a response was received.";

  const wrapAPIs = extensionAPIs => {
    const apiMetadata = {
      "alarms": {
        "clear": { "minArgs": 0, "maxArgs": 1 },
        "clearAll": { "minArgs": 0, "maxArgs": 0 },
        "get": { "minArgs": 0, "maxArgs": 1 },
        "getAll": { "minArgs": 0, "maxArgs": 0 }
      },
      "bookmarks": {
        "create": { "minArgs": 1, "maxArgs": 1 },
        "get": { "minArgs": 1, "maxArgs": 1 },
        "getChildren": { "minArgs": 1, "maxArgs": 1 },
        "getRecent": { "minArgs": 1, "maxArgs": 1 },
        "getSubTree": { "minArgs": 1, "maxArgs": 1 },
        "getTree": { "minArgs": 0, "maxArgs": 0 },
        "move": { "minArgs": 2, "maxArgs": 2 },
        "remove": { "minArgs": 1, "maxArgs": 1 },
        "removeTree": { "minArgs": 1, "maxArgs": 1 },
        "search": { "minArgs": 1, "maxArgs": 1 },
        "update": { "minArgs": 2, "maxArgs": 2 }
      },
      "browserAction": {
        "disable": { "minArgs": 0, "maxArgs": 1, "fallbackToNoCallback": true },
        "enable": { "minArgs": 0, "maxArgs": 1, "fallbackToNoCallback": true },
        "getBadgeBackgroundColor": { "minArgs": 1, "maxArgs": 1 },
        "getBadgeText": { "minArgs": 1, "maxArgs": 1 },
        "getPopup": { "minArgs": 1, "maxArgs": 1 },
        "getTitle": { "minArgs": 1, "maxArgs": 1 },
        "openPopup": { "minArgs": 0, "maxArgs": 0 },
        "setBadgeBackgroundColor": { "minArgs": 1, "maxArgs": 1, "fallbackToNoCallback": true },
        "setBadgeText": { "minArgs": 1, "maxArgs": 1, "fallbackToNoCallback": true },
        "setIcon": { "minArgs": 1, "maxArgs": 1 },
        "setPopup": { "minArgs": 1, "maxArgs": 1, "fallbackToNoCallback": true },
        "setTitle": { "minArgs": 1, "maxArgs": 1, "fallbackToNoCallback": true }
      },
      "browsingData": {
        "remove": { "minArgs": 2, "maxArgs": 2 },
        "removeCache": { "minArgs": 1, "maxArgs": 1 },
        "removeCookies": { "minArgs": 1, "maxArgs": 1 },
        "removeDownloads": { "minArgs": 1, "maxArgs": 1 },
        "removeFormData": { "minArgs": 1, "maxArgs": 1 },
        "removeHistory": { "minArgs": 1, "maxArgs": 1 },
        "removeLocalStorage": { "minArgs": 1, "maxArgs": 1 },
        "removePasswords": { "minArgs": 1, "maxArgs": 1 },
        "removePluginData": { "minArgs": 1, "maxArgs": 1 },
        "settings": { "minArgs": 0, "maxArgs": 0 }
      },
      "commands": {
        "getAll": { "minArgs": 0, "maxArgs": 0 }
      },
      "contextMenus": {
        "remove": { "minArgs": 1, "maxArgs": 1 },
        "removeAll": { "minArgs": 0, "maxArgs": 0 },
        "update": { "minArgs": 2, "maxArgs": 2 }
      },
      "cookies": {
        "get": { "minArgs": 1, "maxArgs": 1 },
        "getAll": { "minArgs": 1, "maxArgs": 1 },
        "getAllCookieStores": { "minArgs": 0, "maxArgs": 0 },
        "remove": { "minArgs": 1, "maxArgs": 1 },
        "set": { "minArgs": 1, "maxArgs": 1 }
      },
      "devtools": {
        "inspectedWindow": {
          "eval": { "minArgs": 1, "maxArgs": 2, "singleCallbackArg": false }
        },
        "panels": {
          "create": { "minArgs": 3, "maxArgs": 3, "singleCallbackArg": true },
          "elements": {
            "createSidebarPane": { "minArgs": 1, "maxArgs": 1 }
          }
        }
      },
      "downloads": {
        "cancel": { "minArgs": 1, "maxArgs": 1 },
        "download": { "minArgs": 1, "maxArgs": 1 },
        "erase": { "minArgs": 1, "maxArgs": 1 },
        "getFileIcon": { "minArgs": 1, "maxArgs": 2 },
        "open": { "minArgs": 1, "maxArgs": 1, "fallbackToNoCallback": true },
        "pause": { "minArgs": 1, "maxArgs": 1 },
        "removeFile": { "minArgs": 1, "maxArgs": 1 },
        "resume": { "minArgs": 1, "maxArgs": 1 },
        "search": { "minArgs": 1, "maxArgs": 1 },
        "show": { "minArgs": 1, "maxArgs": 1, "fallbackToNoCallback": true }
      },
      "extension": {
        "isAllowedFileSchemeAccess": { "minArgs": 0, "maxArgs": 0 },
        "isAllowedIncognitoAccess": { "minArgs": 0, "maxArgs": 0 }
      },
      "history": {
        "addUrl": { "minArgs": 1, "maxArgs": 1 },
        "deleteAll": { "minArgs": 0, "maxArgs": 0 },
        "deleteRange": { "minArgs": 1, "maxArgs": 1 },
        "deleteUrl": { "minArgs": 1, "maxArgs": 1 },
        "getVisits": { "minArgs": 1, "maxArgs": 1 },
        "search": { "minArgs": 1, "maxArgs": 1 }
      },
      "i18n": {
        "detectLanguage": { "minArgs": 1, "maxArgs": 1 },
        "getAcceptLanguages": { "minArgs": 0, "maxArgs": 0 }
      },
      "identity": {
        "launchWebAuthFlow": { "minArgs": 1, "maxArgs": 1 }
      },
      "idle": {
        "queryState": { "minArgs": 1, "maxArgs": 1 }
      },
      "management": {
        "get": { "minArgs": 1, "maxArgs": 1 },
        "getAll": { "minArgs": 0, "maxArgs": 0 },
        "getSelf": { "minArgs": 0, "maxArgs": 0 },
        "setEnabled": { "minArgs": 2, "maxArgs": 2 },
        "uninstallSelf": { "minArgs": 0, "maxArgs": 1 }
      },
      "notifications": {
        "clear": { "minArgs": 1, "maxArgs": 1 },
        "create": { "minArgs": 1, "maxArgs": 2 },
        "getAll": { "minArgs": 0, "maxArgs": 0 },
        "getPermissionLevel": { "minArgs": 0, "maxArgs": 0 },
        "update": { "minArgs": 2, "maxArgs": 2 }
      },
      "pageAction": {
        "getPopup": { "minArgs": 1, "maxArgs": 1 },
        "getTitle": { "minArgs": 1, "maxArgs": 1 },
        "hide": { "minArgs": 1, "maxArgs": 1, "fallbackToNoCallback": true },
        "setIcon": { "minArgs": 1, "maxArgs": 1 },
        "setPopup": { "minArgs": 1, "maxArgs": 1, "fallbackToNoCallback": true },
        "setTitle": { "minArgs": 1, "maxArgs": 1, "fallbackToNoCallback": true },
        "show": { "minArgs": 1, "maxArgs": 1, "fallbackToNoCallback": true }
      },
      "permissions": {
        "contains": { "minArgs": 1, "maxArgs": 1 },
        "getAll": { "minArgs": 0, "maxArgs": 0 },
        "remove": { "minArgs": 1, "maxArgs": 1 },
        "request": { "minArgs": 1, "maxArgs": 1 }
      },
      "runtime": {
        "getBackgroundPage": { "minArgs": 0, "maxArgs": 0 },
        "getPlatformInfo": { "minArgs": 0, "maxArgs": 0 },
        "openOptionsPage": { "minArgs": 0, "maxArgs": 0 },
        "requestUpdateCheck": { "minArgs": 0, "maxArgs": 0 },
        "sendMessage": { "minArgs": 1, "maxArgs": 3 },
        "sendNativeMessage": { "minArgs": 2, "maxArgs": 2 },
        "setUninstallURL": { "minArgs": 1, "maxArgs": 1 }
      },
      "sessions": {
        "getDevices": { "minArgs": 0, "maxArgs": 1 },
        "getRecentlyClosed": { "minArgs": 0, "maxArgs": 1 },
        "restore": { "minArgs": 0, "maxArgs": 1 }
      },
      "storage": {
        "local": {
          "clear": { "minArgs": 0, "maxArgs": 0 },
          "get": { "minArgs": 0, "maxArgs": 1 },
          "getBytesInUse": { "minArgs": 0, "maxArgs": 1 },
          "remove": { "minArgs": 1, "maxArgs": 1 },
          "set": { "minArgs": 1, "maxArgs": 1 }
        },
        "managed": {
          "get": { "minArgs": 0, "maxArgs": 1 },
          "getBytesInUse": { "minArgs": 0, "maxArgs": 1 }
        },
        "sync": {
          "clear": { "minArgs": 0, "maxArgs": 0 },
          "get": { "minArgs": 0, "maxArgs": 1 },
          "getBytesInUse": { "minArgs": 0, "maxArgs": 1 },
          "remove": { "minArgs": 1, "maxArgs": 1 },
          "set": { "minArgs": 1, "maxArgs": 1 }
        }
      },
      "tabs": {
        "captureVisibleTab": { "minArgs": 0, "maxArgs": 2 },
        "create": { "minArgs": 1, "maxArgs": 1 },
        "detectLanguage": { "minArgs": 0, "maxArgs": 1 },
        "discard": { "minArgs": 0, "maxArgs": 1 },
        "duplicate": { "minArgs": 1, "maxArgs": 1 },
        "executeScript": { "minArgs": 1, "maxArgs": 2 },
        "get": { "minArgs": 1, "maxArgs": 1 },
        "getCurrent": { "minArgs": 0, "maxArgs": 0 },
        "getZoom": { "minArgs": 0, "maxArgs": 1 },
        "getZoomSettings": { "minArgs": 0, "maxArgs": 1 },
        "goBack": { "minArgs": 0, "maxArgs": 1 },
        "goForward": { "minArgs": 0, "maxArgs": 1 },
        "highlight": { "minArgs": 1, "maxArgs": 1 },
        "insertCSS": { "minArgs": 1, "maxArgs": 2 },
        "move": { "minArgs": 2, "maxArgs": 2 },
        "query": { "minArgs": 1, "maxArgs": 1 },
        "reload": { "minArgs": 0, "maxArgs": 2 },
        "remove": { "minArgs": 1, "maxArgs": 1 },
        "removeCSS": { "minArgs": 1, "maxArgs": 2 },
        "sendMessage": { "minArgs": 2, "maxArgs": 3 },
        "setZoom": { "minArgs": 1, "maxArgs": 2 },
        "setZoomSettings": { "minArgs": 1, "maxArgs": 2 },
        "update": { "minArgs": 1, "maxArgs": 2 }
      },
      "topSites": {
        "get": { "minArgs": 0, "maxArgs": 0 }
      },
      "webNavigation": {
        "getAllFrames": { "minArgs": 1, "maxArgs": 1 },
        "getFrame": { "minArgs": 1, "maxArgs": 1 }
      },
      "webRequest": {
        "handlerBehaviorChanged": { "minArgs": 0, "maxArgs": 0 }
      },
      "windows": {
        "create": { "minArgs": 0, "maxArgs": 1 },
        "get": { "minArgs": 1, "maxArgs": 2 },
        "getAll": { "minArgs": 0, "maxArgs": 1 },
        "getCurrent": { "minArgs": 0, "maxArgs": 1 },
        "getLastFocused": { "minArgs": 0, "maxArgs": 1 },
        "remove": { "minArgs": 1, "maxArgs": 1 },
        "update": { "minArgs": 2, "maxArgs": 2 }
      }
    };

    if (Object.keys(apiMetadata).length === 0) {
      throw new Error("api-metadata.json has not been included in browser-polyfill");
    }

    class DefaultWeakMap extends WeakMap {
      constructor(createItem, items = undefined) {
        super(items);
        this.createItem = createItem;
      }
      get(key) {
        if (!this.has(key)) {
          this.set(key, this.createItem(key));
        }
        return super.get(key);
      }
    }

    const isThenable = value => {
      return value && typeof value === "object" && typeof value.then === "function";
    };

    const makeCallback = (promise, metadata) => {
      return (...callbackArgs) => {
        if (extensionAPIs.runtime.lastError) {
          promise.reject(new Error(extensionAPIs.runtime.lastError.message));
        } else if (metadata.singleCallbackArg || callbackArgs.length <= 1 && metadata.singleCallbackArg !== false) {
          promise.resolve(callbackArgs[0]);
        } else {
          promise.resolve(callbackArgs);
        }
      };
    };

    const pluralizeArguments = numArgs => numArgs == 1 ? "argument" : "arguments";

    const wrapAsyncFunction = (name, metadata) => {
      return function asyncFunctionWrapper(target, ...args) {
        if (args.length < metadata.minArgs) {
          throw new Error(`Expected at least ${metadata.minArgs} ${pluralizeArguments(metadata.minArgs)} for ${name}(), got ${args.length}`);
        }
        if (args.length > metadata.maxArgs) {
          throw new Error(`Expected at most ${metadata.maxArgs} ${pluralizeArguments(metadata.maxArgs)} for ${name}(), got ${args.length}`);
        }
        return new Promise((resolve, reject) => {
          if (metadata.fallbackToNoCallback) {
            try {
              target[name](...args, makeCallback({ resolve, reject }, metadata));
            } catch (cbError) {
              console.warn(`${name} API method doesn't seem to support the callback parameter, falling back to call it without a callback: `, cbError);
              target[name](...args);
              metadata.fallbackToNoCallback = false;
              metadata.noCallback = true;
              resolve();
            }
          } else if (metadata.noCallback) {
            target[name](...args);
            resolve();
          } else {
            target[name](...args, makeCallback({ resolve, reject }, metadata));
          }
        });
      };
    };

    const wrapMethod = (target, method, wrapper) => {
      return new Proxy(method, {
        apply(targetMethod, thisObj, args) {
          return wrapper.call(thisObj, target, ...args);
        }
      });
    };

    let hasOwnProperty = Function.call.bind(Object.prototype.hasOwnProperty);

    const wrapObject = (target, wrappers = {}, metadata = {}) => {
      let cache = Object.create(null);
      let handlers = {
        has(proxyTarget, prop) {
          return prop in target || prop in cache;
        },
        get(proxyTarget, prop, receiver) {
          if (prop in cache) {
            return cache[prop];
          }
          if (!(prop in target)) {
            return undefined;
          }
          let value = target[prop];
          if (typeof value === "function") {
            if (typeof wrappers[prop] === "function") {
              value = wrapMethod(target, target[prop], wrappers[prop]);
            } else if (hasOwnProperty(metadata, prop)) {
              let wrapper = wrapAsyncFunction(prop, metadata[prop]);
              value = wrapMethod(target, target[prop], wrapper);
            } else {
              value = value.bind(target);
            }
          } else if (typeof value === "object" && value !== null && (hasOwnProperty(wrappers, prop) || hasOwnProperty(metadata, prop))) {
            value = wrapObject(value, wrappers[prop], metadata[prop]);
          } else if (hasOwnProperty(metadata, "*")) {
            value = wrapObject(value, wrappers[prop], metadata["*"]);
          } else {
            Object.defineProperty(cache, prop, {
              configurable: true,
              enumerable: true,
              get() { return target[prop]; },
              set(value) { target[prop] = value; }
            });
            return value;
          }
          cache[prop] = value;
          return value;
        },
        set(proxyTarget, prop, value, receiver) {
          if (prop in cache) {
            cache[prop] = value;
          } else {
            target[prop] = value;
          }
          return true;
        },
        defineProperty(proxyTarget, prop, desc) {
          return Reflect.defineProperty(cache, prop, desc);
        },
        deleteProperty(proxyTarget, prop) {
          return Reflect.deleteProperty(cache, prop);
        }
      };

      let proxyTarget = Object.create(target);
      return new Proxy(proxyTarget, handlers);
    };

    const wrapEvent = wrapperMap => ({
      addListener(target, listener, ...args) {
        target.addListener(wrapperMap.get(listener), ...args);
      },
      hasListener(target, listener) {
        return target.hasListener(wrapperMap.get(listener));
      },
      removeListener(target, listener) {
        target.removeListener(wrapperMap.get(listener));
      }
    });

    const onRequestFinishedWrappers = new DefaultWeakMap(listener => {
      if (typeof listener !== "function") {
        return listener;
      }
      return function onRequestFinished(req) {
        const wrappedReq = wrapObject(req, {} /* wrappers */, {
          getContent: { minArgs: 0, maxArgs: 0 }
        });
        listener(wrappedReq);
      };
    });

    const onMessageWrappers = new DefaultWeakMap(listener => {
      if (typeof listener !== "function") {
        return listener;
      }
      return function onMessage(message, sender, sendResponse) {
        let didCallSendResponse = false;
        let wrappedSendResponse;
        let sendResponsePromise = new Promise(resolve => {
          wrappedSendResponse = function (response) {
            didCallSendResponse = true;
            resolve(response);
          };
        });
        let result;
        try {
          result = listener(message, sender, wrappedSendResponse);
        } catch (err) {
          result = Promise.reject(err);
        }
        const isResultThenable = result !== true && isThenable(result);
        if (result !== true && !isResultThenable && !didCallSendResponse) {
          return false;
        }
        const sendPromisedResult = promise => {
          promise.then(msg => {
            sendResponse(msg);
          }, error => {
            let message;
            if (error && (error instanceof Error || typeof error.message === "string")) {
              message = error.message;
            } else {
              message = "An unexpected error occurred";
            }
            sendResponse({ __mozWebExtensionPolyfillReject__: true, message });
          }).catch(err => {
            console.error("Failed to send onMessage rejected reply", err);
          });
        };
        if (isResultThenable) {
          sendPromisedResult(result);
        } else {
          sendPromisedResult(sendResponsePromise);
        }
        return true;
      };
    });

    const wrappedSendMessageCallback = ({ reject, resolve }, reply) => {
      if (extensionAPIs.runtime.lastError) {
        if (extensionAPIs.runtime.lastError.message === CHROME_SEND_MESSAGE_CALLBACK_NO_RESPONSE_MESSAGE) {
          resolve();
        } else {
          reject(new Error(extensionAPIs.runtime.lastError.message));
        }
      } else if (reply && reply.__mozWebExtensionPolyfillReject__) {
        reject(new Error(reply.message));
      } else {
        resolve(reply);
      }
    };

    const wrappedSendMessage = (name, metadata, apiNamespaceObj, ...args) => {
      if (args.length < metadata.minArgs) {
        throw new Error(`Expected at least ${metadata.minArgs} ${pluralizeArguments(metadata.minArgs)} for ${name}(), got ${args.length}`);
      }
      if (args.length > metadata.maxArgs) {
        throw new Error(`Expected at most ${metadata.maxArgs} ${pluralizeArguments(metadata.maxArgs)} for ${name}(), got ${args.length}`);
      }
      return new Promise((resolve, reject) => {
        const wrappedCb = wrappedSendMessageCallback.bind(null, { resolve, reject });
        args.push(wrappedCb);
        apiNamespaceObj.sendMessage(...args);
      });
    };

    const staticWrappers = {
      devtools: {
        network: {
          onRequestFinished: wrapEvent(onRequestFinishedWrappers)
        }
      },
      runtime: {
        onMessage: wrapEvent(onMessageWrappers),
        onMessageExternal: wrapEvent(onMessageWrappers),
        sendMessage: wrappedSendMessage.bind(null, "sendMessage", { minArgs: 1, maxArgs: 3 })
      },
      tabs: {
        sendMessage: wrappedSendMessage.bind(null, "sendMessage", { minArgs: 2, maxArgs: 3 })
      }
    };

    const settingMetadata = {
      clear: { minArgs: 1, maxArgs: 1 },
      get: { minArgs: 1, maxArgs: 1 },
      set: { minArgs: 1, maxArgs: 1 }
    };

    apiMetadata.privacy = {
      network: { "*": settingMetadata },
      services: { "*": settingMetadata },
      websites: { "*": settingMetadata }
    };

    return wrapObject(extensionAPIs, staticWrappers, apiMetadata);
  };

  globalThis.browser = wrapAPIs(chrome);
}

function execScript(tabId, func, args = []) {
  return new Promise((resolve, reject) => {
    chrome.scripting.executeScript({
      target: { tabId },
      func,
      args
    }, (results) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
      } else {
        resolve(results.map(r => r.result));
      }
    });
  });
}

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
      contexts: ["selection"]
    });
    browser.contextMenus.create({
      id: "add-page-to-hive",
      title: "Snapshot page to DocuSearch",
      contexts: ["page"]
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
      execScript(tab.id, () => document.body.innerText)
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
      const results = await execScript(tab.id, () => document.body.innerText);
      await addTextDocument(results[0], tab.url, tab.title);
    })()
      .then(() => sendResponse({ success: true }))
      .catch(e => sendResponse({ success: false, error: e.message }));
    return true;
  }

  if (message.type === "INSERT_TEXT") {
    browser.tabs.query({ active: true, currentWindow: true }).then(tabs => {
      const tab = tabs[0];
      execScript(tab.id, (text) => {
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
      }, [message.text]).then(([result]) => sendResponse(result))
        .catch(() => sendResponse({ inserted: false, error: true }));
    });
    return true;
  }

  if (message.type === "GET_SELECTION") {
    browser.tabs.query({ active: true, currentWindow: true }).then(tabs => {
      const tab = tabs[0];
      execScript(tab.id, () => window.getSelection().toString())
        .then(([text]) => sendResponse(text))
        .catch(() => sendResponse(""));
    });
    return true;
  }

  if (message.type === "EXEC_CODE") {
    browser.tabs.query({ active: true, currentWindow: true }).then(tabs => {
      const tab = tabs[0];
      execScript(tab.id, (text) => { navigator.clipboard.writeText(text); }, [message.text])
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

browser.omnibox.onInputEntered.addListener(async (text) => {
  await browser.storage.local.set({ pendingSearch: text });
  const tabs = await browser.tabs.query({ active: true, lastFocusedWindow: true });
  if (tabs && tabs[0]) {
    chrome.sidePanel.open({ windowId: tabs[0].windowId });
  }
});

chrome.action.onClicked.addListener((tab) => {
  chrome.sidePanel.open({ windowId: tab.windowId });
});
