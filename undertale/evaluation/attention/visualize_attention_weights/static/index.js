const link = document.createElement("link");
link.rel = "stylesheet";
link.href = "style.css";
document.head.appendChild(link);

if (!customElements.get("attention-pane")) {
    class AttentionPane extends HTMLElement {
      async connectedCallback() {
      this.innerHTML = `
      <div>
        <div>
          <h1> MLM Inference and Model Attention Visualization </h1>
          <p>
              Input pretokenized and masked [MASK] text and see how the model predicts the missing tokens. This plugin displays the model’s attention weights as an interactive visualization.
          </p>

          <form id="predict_form">
            <textarea id="query" rows="10" cols="60"></textarea>
            <button type="submit">Predict</button>
          </form>
        </div>

        <div style="display:flex; resize: both; overflow: auto; justify-content:center; align-items: center;
    align-items: center;  ">
          <iframe id="frame"></iframe>
        </div>
      </div>
      `;

      const form = this.querySelector("#predict_form");
      const input = this.querySelector("#query");
      const frame = this.querySelector("#frame");
      
      form.addEventListener("submit", (e) => {
        e.preventDefault();

        frame.style.display = 'none';

        const query = encodeURIComponent(input.value.trim());
        frame.src = query ? `attention.html?query=${query}` : "attention.html";

        frame.addEventListener('load', function() {
          frame.style.display = 'flex'; 
        });
      });
    }
  }
  customElements.define("attention-pane", AttentionPane);
}

export function render(root){ const el=document.createElement("attention-pane"); root&&root.appendChild(el); return el; }
export const plugin = { dom(){ return document.createElement("attention-pane"); } };

document.body.appendChild(document.createElement("attention-pane"));