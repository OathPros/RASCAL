const logEl = document.getElementById('chat-log');
const buttonsEl = document.getElementById('action-buttons');
const formEl = document.getElementById('form-container');
let activeActionId = null;
const searchContext = [];

function setActiveActionButton(actionId) {
  activeActionId = actionId;
  const allButtons = buttonsEl.querySelectorAll('button[data-action-id]');
  allButtons.forEach((button) => {
    const isActive = button.dataset.actionId === actionId;
    button.classList.toggle('is-selected', isActive);
    if (!isActive) {
      button.classList.remove('best-match');
    }
  });
}

function addMsg(text, cls='bot') {
  const div = document.createElement('div');
  div.className = `msg ${cls}`;
  div.textContent = text;
  logEl.appendChild(div);
  logEl.scrollTop = logEl.scrollHeight;
}

async function sendMessage() {
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg(`You: ${text}`, 'user');

  const res = await fetch('/api/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text, context: searchContext})
  });
  const data = await res.json();
  if (!res.ok) {
    addMsg(data.error || 'Something went wrong. Please try again.');
    return;
  }

  searchContext.push(text);
  buttonsEl.innerHTML = '';
  addMsg(`Top ${data.candidates.length} matches. Add more details below to refine the search.`);
  addMsg(`Best match: ${data.selected_action_id} (${data.selection_source})`);
  const selected = data.selected_action_id;
  data.candidates.forEach(c => {
    const b = document.createElement('button');
    b.dataset.actionId = c.action_id;
    const actionDescription = (c.action_description_added || c.description || '').trim();
    b.textContent = actionDescription || c.title;
    if (c.action_id === selected) {
      b.classList.add('best-match');
    }
    b.onclick = () => {
      setActiveActionButton(c.action_id);
      loadForm(c.action_id);
    };
    buttonsEl.appendChild(b);
  });
  if (selected) {
    setActiveActionButton(selected);
    loadForm(selected);
  }
}

async function loadForm(actionId) {
  const res = await fetch(`/api/form/${actionId}`);
  const data = await res.json();

  formEl.innerHTML = '<div id="surveyContainer"></div>';
  const model = new Survey.Model({
    title: data.schema.title,
    description: data.schema.description,
    pages: [{ elements: data.schema.elements }],
    completedHtml: '<h4>Submitting...</h4>'
  });

  model.onCompleting.add(async (_, opts) => {
    const submit = await fetch('/api/submit', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ action_id: actionId, answers: model.data })
    });
    const out = await submit.json();
    if (!submit.ok) {
      opts.allowComplete = false;
      alert((out.errors || ['Validation failed']).join('\n'));
      return;
    }
    addMsg(`Submitted. Simulated Halo email routed to ${out.halo_email.to}`);
  });

  model.render('surveyContainer');
}

document.getElementById('send-btn').onclick = sendMessage;
document.getElementById('chat-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') sendMessage();
});
addMsg('Describe your request and I will suggest the top 3 matching service actions. Add more details after the results to refine them.');
