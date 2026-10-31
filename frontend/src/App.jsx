import { useState } from "react";
const API = "http://localhost:8000/api";
export default function App() {
  const [pd, setPd] = useState(null);
  const [form, setForm] = useState({ age:35, income:50000, employment_years:5, credit_history_years:5, num_open_accounts:4, num_times_late:0, loan_amount:10000, loan_type:"cash_loan", loan_term:36, interest_rate:10, dependents:0, has_car:0, has_realty:0, gender:"M" });
  const [tab, setTab] = useState("predict");

  const submit = async () => {
    const res = await fetch(`${API}/predict`, { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(form) });
    const data = await res.json();
    setPd(data);
  };

  const portfolio = async () => {
    const res = await fetch(`${API}/portfolio/summary`);
    setPd(await res.json());
  };

  return (
    <div style={{fontFamily:"system-ui", padding:20, maxWidth:900, margin:"auto"}}>
      <h1>Credit Risk Decisioning</h1>
      <div style={{display:"flex", gap:10, marginBottom:20}}>
        {["predict","portfolio","model","drift"].map(t=>(
          <button key={t} onClick={()=>setTab(t)} style={{padding:"6px 12px", background: tab===t?"#1f3864":"#eee", color: tab===t?"#fff":"#000", border:"none", borderRadius:4}}>{t}</button>
        ))}
      </div>
      {tab==="predict" && (
        <div>
          <h2>Individual Credit Assessment</h2>
          <div style={{display:"grid", gridTemplateColumns:"1fr 1fr", gap:8}}>
            {Object.keys(form).map(k=>(
              <label key={k}>{k}: <input value={form[k]} onChange={e=>setForm({...form,[k]: isNaN(Number(e.target.value))?e.target.value:Number(e.target.value)})} style={{width:"100%"}}/></label>
            ))}
          </div>
          <button onClick={submit} style={{marginTop:12, padding:"8px 16px"}}>Predict</button>
          {pd && <pre style={{background:"#f4f4f4", padding:12, marginTop:12}}>{JSON.stringify(pd,null,2)}</pre>}
        </div>
      )}
      {tab==="portfolio" && <div><h2>Portfolio</h2><button onClick={portfolio}>Load</button>{pd && <pre>{JSON.stringify(pd,null,2)}</pre>}</div>}
      {tab==="model" && <div><h2>Model</h2><button onClick={async()=>{const r=await fetch(`${API}/model/info`); setPd(await r.json());}}>Load</button>{pd && <pre>{JSON.stringify(pd,null,2)}</pre>}</div>}
      {tab==="drift" && <div><h2>Drift</h2><button onClick={async()=>{const r=await fetch(`${API}/drift/check`); setPd(await r.json());}}>Check</button>{pd && <pre>{JSON.stringify(pd,null,2)}</pre>}</div>}
    </div>
  );
}
