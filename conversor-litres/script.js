"use strict";

/* =========================================================================
 * Conversor educatiu a litres
 * -------------------------------------------------------------------------
 * Els càlculs es fan amb nombres decimals exactes (BigInt + escala) per
 * evitar errors de coma flotant (p. ex. 0,1 + 0,2 = 0,30000000000000004).
 * Un decimal es guarda com { m, s } i val m / 10^s.
 * ========================================================================= */

// ---------- Aritmètica decimal exacta ----------

const DEU = 10n;

function normalitza(d) {
  let { m, s } = d;
  while (s > 0 && m % DEU === 0n) { m /= DEU; s--; }
  return { m, s };
}

/** Converteix un text ("33", "0,5", "1.5") en decimal exacte, o null si no és vàlid. */
function llegeixDecimal(text) {
  const net = String(text).trim().replace(/[\s  ]/g, "").replace(",", ".");
  if (!/^\d*\.?\d*$/.test(net) || !/\d/.test(net)) return null;
  const [ent = "", frac = ""] = net.split(".");
  if ((ent + frac).replace(/^0+/, "").length > 15) return null; // massa xifres
  return normalitza({ m: BigInt((ent + frac) || "0"), s: frac.length });
}

/** Multiplica per 10^n (n negatiu = divideix). És "moure la coma". */
function mouComa(d, n) {
  let s = d.s - n, m = d.m;
  if (s < 0) { m *= DEU ** BigInt(-s); s = 0; }
  return normalitza({ m, s });
}

function multiplica(a, b) {
  return normalitza({ m: a.m * b.m, s: a.s + b.s });
}

function iguals(a, b) {
  return a.m === b.m && a.s === b.s;
}

function arrodoneix(d, decimals) {
  if (d.s <= decimals) return d;
  const div = DEU ** BigInt(d.s - decimals);
  return normalitza({ m: (d.m + div / 2n) / div, s: decimals });
}

function aNombre(d) {
  return Number(d.m) / 10 ** d.s;
}

/** Format català: coma decimal i espai fi per als milers (1 000,5). */
function format(d) {
  const xifres = d.m.toString().padStart(d.s + 1, "0");
  const ent = xifres.slice(0, xifres.length - d.s);
  const frac = xifres.slice(xifres.length - d.s);
  const entAgrupat = ent.length >= 4 ? ent.replace(/\B(?=(\d{3})+(?!\d))/g, " ") : ent;
  return frac ? `${entAgrupat},${frac}` : entAgrupat;
}

const dec = (text) => llegeixDecimal(text);

// ---------- Definició de les unitats ----------

// Escala mètrica, de més gran a més petita. L'índex 3 és el litre.
const ESCALA = [
  { id: "kL",  simbol: "kL",  nom: "quilolitre",  plural: "quilolitres",  prefix: "quilo", significat: "mil",                   equivCub: "m³" },
  { id: "hL",  simbol: "hL",  nom: "hectolitre",  plural: "hectolitres",  prefix: "hecto", significat: "cent",                  equivCub: "" },
  { id: "daL", simbol: "daL", nom: "decalitre",   plural: "decalitres",   prefix: "deca",  significat: "deu",                   equivCub: "" },
  { id: "L",   simbol: "L",   nom: "litre",       plural: "litres",       prefix: "",      significat: "la unitat principal",   equivCub: "dm³" },
  { id: "dL",  simbol: "dL",  nom: "decilitre",   plural: "decilitres",   prefix: "deci",  significat: "la desena part",        equivCub: "" },
  { id: "cL",  simbol: "cL",  nom: "centilitre",  plural: "centilitres",  prefix: "centi", significat: "la centèsima part",     equivCub: "" },
  { id: "mL",  simbol: "mL",  nom: "mil·lilitre", plural: "mil·lilitres", prefix: "mil·li", significat: "la mil·lèsima part",   equivCub: "cm³" },
];
const INDEX_LITRE = 3;

const UNITATS = [
  ...ESCALA.map((u, i) => ({ ...u, tipus: "metric", index: i, exponent: INDEX_LITRE - i })),
  { id: "m3",  simbol: "m³",  nom: "metre cúbic",      plural: "metres cúbics",      tipus: "cubic", equival: "kL",
    explicacio: "Un cub d'1 metre de costat. Hi caben 1 000 litres: és el mateix que 1 quilolitre (kL)." },
  { id: "dm3", simbol: "dm³", nom: "decímetre cúbic",  plural: "decímetres cúbics",  tipus: "cubic", equival: "L",
    explicacio: "Un cub de 10 cm de costat. Hi cap exactament 1 litre: per això 1 dm³ = 1 L." },
  { id: "cm3", simbol: "cm³", nom: "centímetre cúbic", plural: "centímetres cúbics", tipus: "cubic", equival: "mL",
    explicacio: "Un cub d'1 cm de costat (com un dau petit). Hi cap 1 mil·lilitre: 1 cm³ = 1 mL. Per això a les xeringues de vegades hi posa «cc»." },
  { id: "gal", simbol: "gal", nom: "galó americà",     plural: "galons americans",   tipus: "altre", factor: "3.785411784",
    explicacio: "Unitat dels Estats Units (p. ex. per a la gasolina). No és del sistema mètric." },
  { id: "pt",  simbol: "pt",  nom: "pinta britànica",  plural: "pintes britàniques", tipus: "altre", factor: "0.56826125",
    explicacio: "Unitat del Regne Unit, famosa per servir-hi la cervesa. No és del sistema mètric." },
  { id: "floz", simbol: "fl oz", nom: "unça líquida americana", plural: "unces líquides americanes", tipus: "altre", factor: "0.0295735295625",
    explicacio: "Unitat dels Estats Units que trobaràs en moltes etiquetes de begudes. No és del sistema mètric." },
];

const PER_ID = Object.fromEntries(UNITATS.map((u) => [u.id, u]));

const EXEMPLES = [
  { text: "🥫 Llauna de 33 cL",       valor: "33",  unitat: "cL" },
  { text: "🥫 Llauna de 500 mL",      valor: "500", unitat: "mL" },
  { text: "💧 Ampolla d'1,5 L",       valor: "1,5", unitat: "L" },
  { text: "🍷 Ampolla de vi de 75 cL", valor: "75",  unitat: "cL" },
  { text: "🥛 Got de 2 dL",           valor: "2",   unitat: "dL" },
  { text: "🥄 Cullereta de 5 mL",     valor: "5",   unitat: "mL" },
  { text: "🪣 Galleda d'1 daL",       valor: "1",   unitat: "daL" },
  { text: "🛢️ Barril de 0,5 hL",      valor: "0,5", unitat: "hL" },
  { text: "💉 Xeringa de 3 cm³",      valor: "3",   unitat: "cm3" },
  { text: "🧊 Cub d'1 dm³",           valor: "1",   unitat: "dm3" },
  { text: "🏊 Dipòsit d'1 m³",        valor: "1",   unitat: "m3" },
  { text: "🍺 Pinta anglesa",         valor: "1",   unitat: "pt" },
  { text: "⛽ Galó americà",          valor: "1",   unitat: "gal" },
];

// ---------- Lògica de conversió ----------

/** Unitat de l'escala mètrica que fem servir per a una unitat (o null si no n'hi ha). */
function unitatEscala(u) {
  if (u.tipus === "metric") return u;
  if (u.tipus === "cubic") return PER_ID[u.equival];
  return null;
}

/** Retorna el resultat en litres i tota la informació per explicar-lo. */
function converteix(valor, u) {
  const escala = unitatEscala(u);
  if (escala) {
    const graons = INDEX_LITRE - escala.index; // > 0: unitat gran (multipliquem); < 0: petita (dividim)
    const cami = [];
    let actual = valor;
    const pas = graons > 0 ? 1 : -1; // direcció cap a l'índex del litre
    for (let i = escala.index; i !== INDEX_LITRE; i += pas) {
      actual = mouComa(actual, graons > 0 ? 1 : -1);
      cami.push({ unitat: ESCALA[i + pas], valor: actual });
    }
    return { litres: mouComa(valor, graons), escala, graons, cami, exacte: true };
  }
  const factor = dec(u.factor);
  return { litres: multiplica(valor, factor), factor, exacte: false };
}

// ---------- Elements del DOM ----------

const $ = (id) => document.getElementById(id);
const inputValor = $("valor");
const selectUnitat = $("unitat");
const error = $("error");

const fraccio = (num, den) =>
  `<span class="fraccio"><span>${num}</span><span>${den}</span></span>`;
const ratllat = (t) => `<span class="ratllat">${t}</span>`;

// ---------- Inicialització ----------

function omplirSelect() {
  const grups = [
    ["Sistema mètric (capacitat)", "metric"],
    ["Unitats cúbiques (volum)", "cubic"],
    ["Altres sistemes", "altre"],
  ];
  for (const [etiqueta, tipus] of grups) {
    const og = document.createElement("optgroup");
    og.label = etiqueta;
    for (const u of UNITATS.filter((x) => x.tipus === tipus)) {
      const opt = document.createElement("option");
      opt.value = u.id;
      opt.textContent = `${u.simbol} — ${u.nom}`;
      og.appendChild(opt);
    }
    selectUnitat.appendChild(og);
  }
  selectUnitat.value = "cL";
}

function omplirExemples() {
  const cont = $("exemples");
  for (const e of EXEMPLES) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "xip";
    b.textContent = e.text;
    b.addEventListener("click", () => {
      inputValor.value = e.valor;
      selectUnitat.value = e.unitat;
      actualitza();
      $("resultat").scrollIntoView({ behavior: "smooth", block: "center" });
    });
    cont.appendChild(b);
  }
}

function omplirEscala() {
  $("escala").innerHTML = ESCALA.map((u, i) => `
    <div class="grao${i === INDEX_LITRE ? " litre" : ""}" data-index="${i}" style="min-height:${70 + (ESCALA.length - i) * 10}px">
      <span class="simbol">${u.simbol}</span>
      <span class="nom">${u.nom}</span>
      ${u.equivCub ? `<span class="nom">= 1 ${u.equivCub}</span>` : ""}
    </div>`).join("") ;
  $("escala").insertAdjacentHTML("afterend", `
    <div class="llegenda">
      <span class="l-origen">unitat de partida</span>
      <span class="l-desti">litre (destí)</span>
      <span class="l-cami">graons que travessem</span>
    </div>`);
}

function omplirTaula() {
  const grups = [
    ["Sistema mètric (capacitat)", "metric"],
    ["Unitats cúbiques (volum)", "cubic"],
    ["Altres sistemes", "altre"],
  ];
  let html = "";
  for (const [etiqueta, tipus] of grups) {
    html += `<tr class="grup"><td colspan="4">${etiqueta}</td></tr>`;
    for (const u of UNITATS.filter((x) => x.tipus === tipus)) {
      const { litres } = converteix(dec("1"), u);
      let sentit;
      if (u.tipus === "metric") {
        sentit = u.prefix ? `«${u.prefix}» = ${u.significat}` : "unitat principal";
      } else if (u.tipus === "cubic") {
        sentit = `1 ${u.simbol} = 1 ${PER_ID[u.equival].simbol}`;
      } else {
        sentit = "equivalència fixa (no decimal)";
      }
      html += `<tr data-id="${u.id}"><td>${u.nom}</td><td><strong>${u.simbol}</strong></td><td>${sentit}</td><td>1 ${u.simbol} = ${format(litres)} L</td></tr>`;
    }
  }
  $("taula-cos").innerHTML = html;
}

// ---------- Actualització principal ----------

function actualitza() {
  const valor = llegeixDecimal(inputValor.value);
  const u = PER_ID[selectUnitat.value];

  if (!valor) {
    inputValor.classList.add("invalid");
    error.hidden = false;
    error.textContent = "Escriu un nombre positiu, per exemple 33 o 1,5 (pots fer servir coma o punt per als decimals).";
    return;
  }
  inputValor.classList.remove("invalid");
  error.hidden = true;

  const r = converteix(valor, u);
  pintaResultat(valor, u, r);
  pintaVisual(r.litres);
  pintaExplicacio(valor, u, r);
  pintaEscala(r);
  document.querySelectorAll("#taula-cos tr[data-id]").forEach((tr) =>
    tr.classList.toggle("actual", tr.dataset.id === u.id));
}

function textLitres(d) {
  const n = aNombre(d);
  return n === 1 ? "litre" : "litres";
}

function pintaResultat(valor, u, r) {
  const nomU = aNombre(valor) === 1 ? u.nom : u.plural;
  let res = `${format(valor)} ${u.simbol} <small>(${nomU})</small> = <span class="num">${format(r.litres)} L</span>`;
  if (!r.exacte && r.litres.s > 3) {
    res = `${format(valor)} ${u.simbol} <small>(${nomU})</small> ≈ <span class="num">${format(arrodoneix(r.litres, 3))} L</span>`;
  }
  $("resultat").innerHTML = res;
}

function pintaVisual(litres) {
  const n = aNombre(litres);
  const visual = $("visual");
  if (n === 0) { visual.innerHTML = ""; return; }
  const MAX = 10;
  const ampolles = Math.min(Math.ceil(n), MAX);
  let html = "";
  for (let i = 0; i < ampolles; i++) {
    const ple = Math.min(1, n - i);
    html += `<div class="ampolla" title="Ampolla d'1 L"><div class="liquid" style="height:${Math.max(ple * 100, 2)}%"></div></div>`;
  }
  let nota = "Cada ampolla dibuixada és d'1 litre.";
  if (n > MAX) nota += ` (N'hi caldrien ${Math.ceil(n).toLocaleString("ca")}; només en dibuixem ${MAX}.)`;
  if (n < 0.02) nota += " La quantitat és tan petita que gairebé no es veu!";
  visual.innerHTML = html + `<p class="visual-nota">${nota}</p>`;
}

function pintaEscala(r) {
  const origen = r.escala ? r.escala.index : -1;
  document.querySelectorAll(".grao").forEach((g) => {
    const i = Number(g.dataset.index);
    g.classList.toggle("origen", i === origen && origen !== INDEX_LITRE);
    g.classList.toggle("desti", i === INDEX_LITRE);
    const dins = origen >= 0 && i !== origen &&
      ((i > Math.min(origen, INDEX_LITRE) && i < Math.max(origen, INDEX_LITRE)));
    g.classList.toggle("cami", dins);
  });
}

// ---------- Explicació pas a pas ----------

function pintaExplicacio(valor, u, r) {
  const passos = [];
  const V = format(valor);
  const R = format(r.litres);

  // Pas 1: què vol dir la unitat
  if (u.tipus === "metric") {
    passos.push(pasSignificat(u));
  } else if (u.tipus === "cubic") {
    const e = r.escala;
    passos.push({
      titol: `Què és un ${u.nom} (${u.simbol})?`,
      cos: `<p>${u.explicacio}</p>
            <p>Per tant, primer canviem la unitat de volum per la de capacitat equivalent:</p>
            <div class="formula">${V} ${u.simbol} = ${V} ${e.simbol}</div>
            ${e.index !== INDEX_LITRE ? `<p>Ara només cal passar de ${e.simbol} a litres.</p>` : "<p>I ja tenim el resultat en litres!</p>"}`,
    });
  } else {
    passos.push({
      titol: `Què és un ${u.nom} (${u.simbol})?`,
      cos: `<p>${u.explicacio}</p>
            <p>Com que no és del sistema mètric, <strong>no</strong> podem fer servir l'escala de 10 en 10. Cal conèixer la seva equivalència:</p>
            <div class="formula">1 ${u.simbol} = ${format(r.factor)} L</div>`,
    });
  }

  // Pas 2 i 3: els dos mètodes
  if (r.escala && r.graons !== 0) {
    passos.push(pasEscala(valor, r));
    passos.push(pasFactor(valor, r));
  } else if (!r.escala) {
    const F = format(r.factor);
    passos.push({
      titol: "Multipliquem per l'equivalència (factor de conversió)",
      cos: `<p>Cada ${u.simbol} són ${F} L, així que multipliquem la quantitat per ${F}. Les unitats ${u.simbol} es cancel·len i queden litres:</p>
            <div class="formula">${V} ${ratllat(u.simbol)} × ${fraccio(`${F} L`, `1 ${ratllat(u.simbol)}`)} = ${V} × ${F} L = <strong>${R} L</strong></div>
            ${r.litres.s > 3 ? `<p>Arrodonit a mil·lilitres (3 decimals): <strong>${format(arrodoneix(r.litres, 3))} L</strong>.</p>` : ""}`,
    });
  } else if (u.tipus === "metric") {
    passos.push({
      titol: "No cal fer cap càlcul",
      cos: `<p>La quantitat ja està en litres: <strong>${V} L = ${R} L</strong>. 😉 Prova amb una altra unitat, com cL o mL!</p>`,
    });
  }

  // Últim pas: comprovar que té sentit
  passos.push(pasComprovacio(valor, u, r));

  $("explicacio").innerHTML = `<ol class="passos">${passos
    .map((p) => `<li><h3>${p.titol}</h3>${p.cos}</li>`)
    .join("")}</ol>`;
}

function pasSignificat(u) {
  if (u.index === INDEX_LITRE) {
    return {
      titol: "Què és el litre (L)?",
      cos: "<p>El <strong>litre</strong> és la unitat principal de capacitat. Totes les altres unitats del sistema mètric es formen afegint-hi un prefix davant.</p>",
    };
  }
  const un = dec("1");
  const enLitres = format(mouComa(un, u.exponent));
  const fr = u.exponent < 0 ? ` = ${fraccio("1", format(mouComa(un, -u.exponent)))} L` : "";
  return {
    titol: `Què vol dir «${u.nom}» (${u.simbol})?`,
    cos: `<p>La paraula es forma amb el prefix <strong>«${u.prefix}»</strong>, que vol dir <strong>${u.significat}</strong>, més «litre».</p>
          <div class="formula">1 ${u.simbol} = ${u.exponent > 0 ? `${enLitres} L` : `${u.significat} d'un litre${fr} = ${enLitres} L`}</div>`,
  };
}

function pasEscala(valor, r) {
  const { escala, graons, cami } = r;
  const n = Math.abs(graons);
  const puja = graons < 0; // unitat petita → pugem cap al litre → dividim
  const op = puja ? "÷ 10" : "× 10";
  const potencia = format(mouComa(dec("1"), n));

  let cadena = `<span class="valor">${format(valor)} ${escala.simbol}</span>`;
  cami.forEach((c, i) => {
    const final = i === cami.length - 1;
    cadena += `<span class="fletxa">${op}<br>→</span><span class="valor${final ? " final" : ""}">${format(c.valor)} ${c.unitat.simbol}</span>`;
  });

  return {
    titol: "Mètode 1: l'escala (moure la coma)",
    cos: `<p>A l'escala, de <strong>${escala.simbol}</strong> a <strong>L</strong> hi ha <strong>${n} ${n === 1 ? "graó" : "graons"}</strong> cap ${puja ? "amunt" : "avall"}.
          ${puja
            ? `Anem cap a una unitat <strong>més gran</strong>, així que <strong>dividim</strong> per 10 a cada graó.`
            : `Anem cap a una unitat <strong>més petita</strong>, així que <strong>multipliquem</strong> per 10 a cada graó.`}</p>
          <div class="cadena">${cadena}</div>
          <p>${n} ${n === 1 ? "graó" : "graons"} = ${puja ? "dividir" : "multiplicar"} per <strong>${potencia}</strong>, que és el mateix que
          moure la coma <strong>${n} ${n === 1 ? "lloc" : "llocs"} cap a la ${puja ? "esquerra" : "dreta"}</strong>
          ${puja ? "(si falten xifres, hi afegim zeros a l'esquerra)" : "(si falten xifres, hi afegim zeros a la dreta)"}.</p>`,
  };
}

function pasFactor(valor, r) {
  const { escala, graons } = r;
  const S = escala.simbol;
  const potencia = format(mouComa(dec("1"), Math.abs(graons)));
  const V = format(valor);
  const R = format(r.litres);
  const fr = graons < 0
    ? fraccio("1 L", `${potencia} ${ratllat(S)}`)
    : fraccio(`${potencia} L`, `1 ${ratllat(S)}`);
  const operacio = graons < 0 ? `${V} ÷ ${potencia}` : `${V} × ${potencia}`;
  const relacio = graons < 0 ? `1 L = ${potencia} ${S}` : `1 ${S} = ${potencia} L`;

  return {
    titol: "Mètode 2: el factor de conversió",
    cos: `<p>Sabem que <strong>${relacio}</strong>. Escrivim aquesta igualtat com una fracció que val 1, posant a sota la unitat que volem eliminar (${S}).
          Així, en multiplicar, els <strong>${S}</strong> es cancel·len i ens queden <strong>litres</strong>:</p>
          <div class="formula">${V} ${ratllat(S)} × ${fr} = ${operacio} L = <strong>${R} L</strong></div>
          <p>Tots dos mètodes donen el mateix resultat. 🎉</p>`,
  };
}

function pasComprovacio(valor, u, r) {
  const v = aNombre(valor);
  const l = aNombre(r.litres);
  const V = format(valor);
  const R = format(r.litres);
  let raonament;

  if (v === 0) {
    raonament = "<p>Zero de qualsevol unitat és zero litres. No hi ha res a omplir!</p>";
  } else {
    const unLitre = converteix(dec("1"), u).litres;
    const mesPetita = aNombre(unLitre) < 1;
    const igual = iguals(unLitre, dec("1"));
    if (igual) {
      raonament = `<p>1 ${u.simbol} és exactament 1 litre, per tant el número no canvia: ${V} → ${R}. ✔️</p>`;
    } else if (mesPetita) {
      raonament = `<p>El ${u.nom} és una unitat <strong>més petita</strong> que el litre: en caben moltes dins d'un litre.
        Per això, en litres, el número ha de ser <strong>més petit</strong>: ${V} → ${R}. ✔️</p>`;
    } else {
      raonament = `<p>El ${u.nom} és una unitat <strong>més gran</strong> que el litre: cada ${u.simbol} conté diversos litres.
        Per això, en litres, el número ha de ser <strong>més gran</strong>: ${V} → ${R}. ✔️</p>`;
    }

    let comparacio;
    if (l < 1) {
      const vegades = 1 / l;
      const arr = vegades >= 10 ? Math.round(vegades) : Math.round(vegades * 10) / 10;
      comparacio = `Amb una ampolla d'1 litre podries omplir aquesta quantitat unes <strong>${arr.toLocaleString("ca")} vegades</strong>.`;
    } else if (l === 1) {
      comparacio = "És exactament una ampolla d'1 litre.";
    } else {
      comparacio = `És com omplir <strong>${(Math.round(l * 100) / 100).toLocaleString("ca")} ampolles d'1 litre</strong>.`;
    }
    raonament += `<p>📏 Per fer-te'n una idea: ${comparacio}</p>`;
  }

  return { titol: "Comprova que el resultat té sentit", cos: raonament };
}

// ---------- Repte (pràctica) ----------

const VALORS_REPTE = {
  kL: ["1", "2", "0,5", "1,2"],
  hL: ["1", "0,5", "2", "3"],
  daL: ["1", "2", "5", "0,5"],
  dL: ["2", "5", "15", "3", "25"],
  cL: ["33", "25", "50", "75", "150", "5"],
  mL: ["250", "330", "500", "750", "1500", "5", "20"],
  m3: ["1", "2", "0,5"],
  dm3: ["2", "5", "0,75", "3"],
  cm3: ["100", "250", "1000", "5"],
};

const repte = { actual: null, encerts: 0, intents: 0 };

function nouRepte() {
  const ids = Object.keys(VALORS_REPTE).filter((id) => !repte.actual || id !== repte.actual.u.id);
  const id = ids[Math.floor(Math.random() * ids.length)];
  const vals = VALORS_REPTE[id];
  const valor = dec(vals[Math.floor(Math.random() * vals.length)]);
  const u = PER_ID[id];
  repte.actual = { u, valor, r: converteix(valor, u) };
  $("repte-pregunta").innerHTML = `Quants litres són <strong>${format(valor)} ${u.simbol}</strong> (${aNombre(valor) === 1 ? u.nom : u.plural})?`;
  $("repte-resposta").value = "";
  $("repte-feedback").innerHTML = "";
  $("repte-feedback").className = "repte-feedback";
}

function comprovaRepte(ev) {
  ev.preventDefault();
  const { u, valor, r } = repte.actual;
  const resposta = llegeixDecimal($("repte-resposta").value);
  const fb = $("repte-feedback");
  if (!resposta) {
    fb.className = "repte-feedback malament";
    fb.innerHTML = "<p>Escriu un nombre (pots fer servir coma decimal, per exemple 0,33).</p>";
    return;
  }
  repte.intents++;
  const correcte = iguals(resposta, r.litres);
  if (correcte) repte.encerts++;

  const e = r.escala;
  const n = Math.abs(r.graons);
  const pista = n === 0
    ? `1 ${u.simbol} = 1 L, així que el número no canvia.`
    : `${u.tipus === "cubic" ? `Primer: ${u.simbol} = ${e.simbol}. ` : ""}De ${e.simbol} a L hi ha ${n} ${n === 1 ? "graó" : "graons"} cap ${r.graons < 0 ? "amunt → divideix" : "avall → multiplica"} per ${format(mouComa(dec("1"), n))} (mou la coma ${n} ${n === 1 ? "lloc" : "llocs"} a la ${r.graons < 0 ? "esquerra" : "dreta"}).`;

  fb.className = `repte-feedback ${correcte ? "be" : "malament"}`;
  fb.innerHTML = correcte
    ? `<p>✅ <strong>Correcte!</strong> ${format(valor)} ${u.simbol} = ${format(r.litres)} L.</p><p>${pista}</p>`
    : `<p>❌ No és correcte. Has respost ${format(resposta)} L, però la resposta és <strong>${format(r.litres)} L</strong>.</p><p>💡 ${pista}</p>`;

  $("encerts").textContent = repte.encerts;
  $("intents").textContent = repte.intents;
}

// ---------- Arrencada ----------

omplirSelect();
omplirExemples();
omplirEscala();
omplirTaula();
inputValor.addEventListener("input", actualitza);
selectUnitat.addEventListener("change", actualitza);
$("formulari").addEventListener("submit", (e) => e.preventDefault());
$("repte-form").addEventListener("submit", comprovaRepte);
$("repte-nou").addEventListener("click", nouRepte);
actualitza();
nouRepte();
