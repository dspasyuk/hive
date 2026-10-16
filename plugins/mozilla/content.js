browser.runtime.onMessage.addListener((message) => {
  switch (message.type) {
    case "INSERT_TEXT":
      return Promise.resolve(insertIntoPage(message.text));

    case "GET_SELECTION":
      return Promise.resolve(window.getSelection().toString());

    case "COPY_TEXT":
      return copyToClipboard(message.text).then(() => true);
  }
});

function insertIntoPage(text) {
  const el = document.activeElement;

  if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA")) {
    const start = el.selectionStart;
    const end = el.selectionEnd;
    el.value = el.value.slice(0, start) + text + el.value.slice(end);
    const pos = start + text.length;
    el.setSelectionRange(pos, pos);
    el.dispatchEvent(new Event("input", { bubbles: true }));
    el.dispatchEvent(new Event("change", { bubbles: true }));
    return { inserted: true };
  }

  if (el && el.contentEditable === "true") {
    el.focus();
    if (document.execCommand) {
      document.execCommand("insertText", false, text);
      return { inserted: true };
    }
  }

  return { inserted: false };
}

async function copyToClipboard(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
    } else {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    return true;
  } catch (e) {
    console.error("Failed to copy:", e);
    return false;
  }
}
