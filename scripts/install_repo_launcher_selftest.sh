#!/usr/bin/env bash
set -uo pipefail

# Self-test for scripts/install_repo_launcher.sh. Every suite runs against a
# throwaway HOME under a temp dir; nothing here touches your real dotfiles.
#
#   scripts/install_repo_launcher_selftest.sh          # A + P (fast, offline)
#   scripts/install_repo_launcher_selftest.sh --all    # also B, which runs real `uv sync`
#
# A - launcher/alias/PATH semantics against a stub `uv` and stub pythons, so it
#     can assert argv, cwd, exit propagation and "wrote nothing" without network.
# B - the move-a-repo scenario with REAL uv on a minimal editable package. This
#     is the suite that caught the health check resolving mtplx from the caller's
#     cwd and blessing a broken venv; no fake reproduced it.
# P - the rc PATH line: absent / already in front / appended-behind / commented.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/scripts/install_repo_launcher.sh"
[ -f "$SRC" ] || { printf 'expected %s\n' "$SRC" >&2; exit 1; }

RUNDIR=$(mktemp -d "${TMPDIR:-/tmp}/mtplx-selftest.XXXXXX")
# $TMPDIR arrives with a trailing slash, so mktemp hands back a path containing
# `//`, and /var is a symlink to /private/var. The launcher normalizes both through
# `cd && pwd`, so the suites have to compare against the same physical form or every
# path assertion lies.
RUNDIR=$(cd "$RUNDIR" && pwd -P)
RESULT_DIR="$RUNDIR"
trap 'rm -rf "$RUNDIR"' EXIT

with_real=0
for arg in "$@"; do
  case "$arg" in
    --all) with_real=1 ;;
    -h|--help) sed -n '3,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) printf 'unknown argument: %s\n' "$arg" >&2; exit 2 ;;
  esac
done

suite_A() {
T=$RUNDIR/A; rm -rf $T; mkdir -p $T/home/.mtplx $T/uvbin
H=$T/healthy; export UVLOG=$T/uvargs HEALTHFILE=$H
P=0;F=0; ok(){ echo "  OK   $1"; P=$((P+1)); }; no(){ echo "  FAIL $1"; F=$((F+1)); }
mkrepo(){ local r=$1; mkdir -p $r/.venv/bin $r/scripts
  printf '#!/bin/sh\necho "mtplx-fixture 1.0"\n' > $r/.venv/bin/mtplx; chmod +x $r/.venv/bin/mtplx
  cat > $r/.venv/bin/python3 <<SHIM
case "\$*" in
  *mlx.core*) echo "0.9.9-shim"; exit 0 ;;
  *import*mtplx*) [ -f "$H" ] && exit 0 || exit 1 ;;
esac
exit 0
SHIM
  chmod +x $r/.venv/bin/python3; cp "$SRC" $r/scripts/install_repo_launcher.sh; }
printf '#!/bin/sh\nprintf "%%s|%%s\\n" "$(pwd)" "$*" >> "$UVLOG"\n[ "${UV_FAIL:-0}" = 1 ] && exit 3\n[ "${UV_REPAIRS:-0}" = 1 ] && touch "$HEALTHFILE"\nexit 0\n' > $T/uvbin/uv
chmod +x $T/uvbin/uv; export PATH=$T/uvbin:$PATH

echo "=== A1..A2 stato sano: idempotenza, nessun uv, provenienza ==="
: > $H; mkrepo $T/repo
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo/scripts/install_repo_launcher.sh 2>&1); rc=$?
{ [ $rc = 0 ] && echo "$out" | grep -q 'venv: serves'; } && ok "exit 0 + venv sano" || no "rc=$rc :: $out"
[ ! -e $UVLOG ] && ok "uv non invocato" || no "invocato: $(cat $UVLOG)"
grep -q "^# repo: $T/repo$" $T/home/.mtplx/bin/mtplx && ok "provenienza scritta" || no "assente"
L=$(shasum $T/home/.mtplx/bin/mtplx|cut -d' ' -f1); Z=$(shasum $T/home/.zshrc|cut -d' ' -f1)
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo/scripts/install_repo_launcher.sh 2>&1)
echo "$out" | grep -q 'already installed for this checkout' && ok "rerun riconosce il proprio checkout" || no "testo: $out"
echo "$out" | grep -q 'The alias already exists' && ok "The alias already exists" || no "alias"
{ [ "$L" = "$(shasum $T/home/.mtplx/bin/mtplx|cut -d' ' -f1)" ] && [ "$Z" = "$(shasum $T/home/.zshrc|cut -d' ' -f1)" ]; } && ok "byte invariati" || no "toccati"

echo "=== A3 healfcheck: shebang morto => venv rotto (anche con import ok) ==="
: > $H; rm -f $T/home/.mtplx/bin/mtplx
printf '#!/bin/sh\n' > $T/repo/.venv/bin/mtplx; chmod +x $T/repo/.venv/bin/mtplx
printf '#!/no/such/interpreter\n' > $T/repo/.venv/bin/mtplx
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo/scripts/install_repo_launcher.sh --check 2>&1)
echo "$out" | grep -q 'needs uv sync' && ok "shebang del console script ispezionato" || no "non rilevato :: $out"
printf '#!/bin/sh\necho ok\n' > $T/repo/.venv/bin/mtplx; chmod +x $T/repo/.venv/bin/mtplx

echo "=== A3b sync che ripara / sync che non basta ==="
rm -f $H $UVLOG
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx UV_REPAIRS=1 bash $T/repo/scripts/install_repo_launcher.sh 2>&1); rc=$?
{ [ $rc = 0 ] && echo "$out" | grep -q 'venv: repaired'; } && ok "sync che ripara -> exit 0 + conferma" || no "rc=$rc :: $out"
grep -q "$T/repo|sync" $UVLOG && ok "cwd=repo, argv=sync, una chiamata ($(wc -l < $UVLOG|tr -d ' '))" || no "argv: $(cat $UVLOG)"
rm -f $H $UVLOG $T/home/.mtplx/bin/mtplx
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo/scripts/install_repo_launcher.sh 2>&1); rc=$?
{ [ $rc = 1 ] && echo "$out" | grep -q 'still does not import'; } && ok "sync insufficiente -> exit 1 forte" || no "rc=$rc :: $out"
[ ! -e $T/home/.mtplx/bin/mtplx ] && ok "nessun launcher su venv rotto (ordine)" || no "creato"
: > $H

echo "=== A4 sync che fallisce / A5 --no-sync ==="
rm -f $UVLOG; out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx UV_FAIL=1 bash -c "rm -f $H; HOME=$T/home MTPLX_HOME=$T/home/.mtplx UV_FAIL=1 bash $T/repo/scripts/install_repo_launcher.sh 2>&1"); rc=$?
[ $rc = 1 ] && ok "sync fallito -> exit 1" || no "rc=$rc :: $out"
rm -f $H $UVLOG
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo/scripts/install_repo_launcher.sh --no-sync 2>&1); rc=$?
{ [ $rc = 1 ] && [ ! -e $UVLOG ]; } && ok "--no-sync: segnala e non tocca uv" || no "rc=$rc :: $out"
: > $H

echo "=== A6 --check allineato: exit 0, zero scritture ==="
HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo/scripts/install_repo_launcher.sh >/dev/null 2>&1
L=$(shasum $T/home/.mtplx/bin/mtplx|cut -d' ' -f1); Z=$(shasum $T/home/.zshrc|cut -d' ' -f1); rm -f $UVLOG
HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo/scripts/install_repo_launcher.sh --check >/dev/null 2>&1; [ $? = 0 ] && ok "exit 0 su pulito" || no "non 0"
{ [ "$L" = "$(shasum $T/home/.mtplx/bin/mtplx|cut -d' ' -f1)" ] && [ "$Z" = "$(shasum $T/home/.zshrc|cut -d' ' -f1)" ] && [ ! -e $UVLOG ]; } && ok "--check non scrive e non sync-a" || no "effetti collaterali"

echo "=== A7 repo spostato: --check nomina il path morto, exit 1 ==="
mv $T/repo $T/repo2; rm -f $UVLOG
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo2/scripts/install_repo_launcher.sh --check 2>&1); rc=$?
{ [ $rc = 1 ] && echo "$out" | grep -q "generated for $T/repo,"; } && ok "stale rilevato e nominato" || no "rc=$rc :: $out"
[ ! -e $UVLOG ] && ok "nessun sync in read-only" || no "sync!"

echo "=== A8 riparazione automatica del drift ==="
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo2/scripts/install_repo_launcher.sh 2>&1); rc=$?
{ [ $rc = 0 ] && grep -q "^# repo: $T/repo2$" $T/home/.mtplx/bin/mtplx; } && ok "ripuntato al nuovo checkout" || no "rc=$rc :: $out"
echo "$out" | grep -q 'repointed at' && ok "lo dichiara" || no "silenzioso"
HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo2/scripts/install_repo_launcher.sh --check >/dev/null 2>&1; [ $? = 0 ] && ok "--check pulito dopo" || no "drift residuo"

echo "=== A9 due checkout vispi: non decide lui ==="
cp -R $T/repo2 $T/repo3
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo3/scripts/install_repo_launcher.sh 2>&1)
echo "$out" | grep -q 'both still exist' && ok "due checkout vispi rilevati" || no "non rilevato :: $out"
grep -q "^# repo: $T/repo2$" $T/home/.mtplx/bin/mtplx && ok "nessuna scelta silenziosa" || no "sovrascritto"
echo "$out" | grep -q -- '--force' && ok "indica l'uscita esplicita" || no "manca"

echo "=== A10 foreign con target morto ==="
printf '#!/bin/sh\nexec %s/dead/.venv/bin/mtplx "$@"\n' "$T" > $T/home/.mtplx/bin/mtplx
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo2/scripts/install_repo_launcher.sh --check 2>&1); rc=$?
{ [ $rc = 1 ] && echo "$out" | grep -q "it execs: $T/dead/.venv/bin/mtplx"; } && ok "target estrapolato + drift segnalato" || no "rc=$rc :: $out"
grep -q 'dead/.venv' $T/home/.mtplx/bin/mtplx && ok "foreign intatto" || no "manomesso"
printf '#!/bin/sh\nexec %s/repo2/.venv/bin/mtplx "$@"\n' "$T" > $T/home/.mtplx/bin/mtplx
out=$(HOME=$T/home MTPLX_HOME=$T/home/.mtplx bash $T/repo2/scripts/install_repo_launcher.sh --check 2>&1); rc=$?
[ $rc = 0 ] && ok "foreign sano -> non lo disturba (exit 0)" || no "rc=$rc :: $out"

echo "=== A11 invocazione dimenticata: senza override HOME, non deve scrivere ==="
RH=$(shasum ~/.zshrc 2>/dev/null | cut -d' ' -f1); RL=$(shasum ~/.mtplx/bin/mtplx 2>/dev/null | cut -d' ' -f1)
bash $T/repo2/scripts/install_repo_launcher.sh --check >/dev/null 2>&1
[ "$RH" = "$(shasum ~/.zshrc | cut -d' ' -f1)" ] && ok "~/.zshrc reale intatta" || no "SCRITTA sulla HOME vera"
[ "$RL" = "$(shasum ~/.mtplx/bin/mtplx | cut -d' ' -f1)" ] && ok "launcher reale intatto" || no "launcher reale toccato"

LC_ALL=C grep -q '[^ -~]' "$SRC" && no "non-ASCII" || ok "solo ASCII"
bash -n "$SRC" && ok "bash -n" || no "sintassi"
printf "%s
" "PASS=$P FAIL=$F" > "$RESULT_DIR/A-mechanism.result"; rm -rf $T
}

suite_P() {
T=$RUNDIR/P; rm -rf $T; mkdir -p $T
P=0;F=0; ok(){ echo "  OK   $1"; P=$((P+1)); }; no(){ echo "  FAIL $1"; F=$((F+1)); }
mkrepo(){ local r=$1; mkdir -p $r/.venv/bin $r/scripts
  printf '#!/bin/sh\necho fixture\n' > $r/.venv/bin/mtplx; chmod +x $r/.venv/bin/mtplx
  cat > $r/.venv/bin/python3 <<'SH'
case "$*" in
  *mlx.core*) echo "0.9.9-shim"; exit 0 ;;
  *import*mtplx*) exit 0 ;;
esac
exit 0
SH
  chmod +x $r/.venv/bin/python3; cp "$SRC" $r/scripts/install_repo_launcher.sh; }
mkrepo $T/repo
run(){ local h=$1 mh=${2:-$1/.mtplx} rc=${3:-$1/.zshrc}; shift $#; HOME=$h MTPLX_HOME="$mh" MTPLX_RC="$rc" bash $T/repo/scripts/install_repo_launcher.sh; }
runf(){ local h=$1 mh=${2:-$1/.mtplx} rc=${3:-$1/.zshrc}; shift 3; HOME=$h MTPLX_HOME="$mh" MTPLX_RC="$rc" bash $T/repo/scripts/install_repo_launcher.sh "$@"; }

echo "=== P1 rc assente -> PATH + alias creati, una sola occorrenza ==="
mkdir -p $T/h1
out=$(run $T/h1); rc=$?
echo "$out" | grep -q 'PATH: added export PATH="\$HOME/.mtplx/bin:\$PATH"' && ok "aggiunta la forma 'in testa'" || no "rc=$rc :: $out"
[ "$(grep -c 'mtplx/bin:\$PATH' $T/h1/.zshrc)" = 1 ] && ok "una sola riga PATH" || no "dup: $(grep -c . $T/h1/.zshrc)"
grep -q '^alias mtplx=' $T/h1/.zshrc && ok "alias nello stesso blocco, un solo header" || no "alias mancante"
[ "$(grep -c '# Added by scripts/install_repo_launcher.sh' $T/h1/.zshrc)" = 1 ] && ok "header scritto una volta sola" || no "header duplicato"

echo "=== P2 rerun: nulla duplicato ==="
Z=$(shasum $T/h1/.zshrc|cut -d' ' -f1)
out=$(run $T/h1)
echo "$out" | grep -q 'PATH: .mtplx/bin in front' && ok "rilevata come gia' in testa" || no "$out"
echo "$out" | grep -q 'The alias already exists' && ok "alias: messaggio letterale invariato" || no "alias regredito"
[ "$Z" = "$(shasum $T/h1/.zshrc|cut -d' ' -f1)" ] && ok "rc byte-identica" || no "riscritta"

echo "=== P3 IL TUO CASO: riga dell'app, con il suo commento sopra ==="
mkdir -p $T/h3
printf '# === END LOCAL AI CODING SETUP ===\n\n# Added by MTPLX.app — terminal command\nexport PATH="$HOME/.mtplx/bin:$PATH"\n' > $T/h3/.zshrc
out=$(run $T/h3)
echo "$out" | grep -q 'PATH: .mtplx/bin in front' && ok "la riga dell'app conta come sufficiente (nessun duplicato)" || no "non riconosciuta :: $out"
echo "$out" | grep -q 'written by: # Added by MTPLX.app' && ok "dice CHI l'ha scritta (provenienza visibile)" || no "owner non riportato :: $out"
[ "$(grep -c 'mtplx/bin:\$PATH' $T/h3/.zshrc)" = 1 ] && ok "ancora una sola riga" || no "duplicata"

echo "=== P4 forma 'dietro':persa contro brew, deve segnalarla ==="
mkdir -p $T/h4
printf 'export PATH="$PATH:$HOME/.mtplx/bin"\n' > $T/h4/.zshrc
out=$(runf $T/h4 "" "" --check); rc=$?
{ [ $rc = 1 ] && echo "$out" | grep -q 'NOT in front'; } && ok "--check: dietro => exit 1" || no "rc=$rc :: $out"
grep -q '^export PATH="\$PATH' $T/h4/.zshrc && ok "--check non ha toccato il file" || no "scritto in read-only"
out=$(runf $T/h4 "" "")
echo "$out" | grep -q 'the front form gets added' && ok "spiega perche' aggiunge" || no "silenzioso :: $out"
{ grep -q '^export PATH="\$PATH' $T/h4/.zshrc && grep -q '^export PATH="\$HOME/.mtplx/bin:\$PATH"' $T/h4/.zshrc; } && ok "vecchia lasciata, nuova in testa aggiunta" || no "gestione sbagliata del file"
[ "$(grep -c 'mtplx/bin:\$PATH' $T/h4/.zshrc)" = 1 ] && ok "la nuova forma e' unica" || no "troppe front-form"

echo "=== P5 MTPLX_HOME fuori da \$HOME -> path letterale ==="
mkdir -p $T/h5
out=$(run $T/h5 $T/custom-bin); rc=$?
echo "$out" | grep -qF "export PATH=\"$T/custom-bin/bin:\$PATH\"" && ok "usa il path letterale, non \$HOME/..." || no "rc=$rc :: $out"
out=$(runf $T/h5 $T/custom-bin $T/h5/.zshrc --check)
echo "$out" | grep -q 'in front' && ok "la riconosce al secondo giro (idempotente)" || no "non idempotente :: $out"
[ -e $T/h5/.mtplx ] && no "ha creato ~/.mtplx inutile" || ok "nessuna cartella di default creata per errore"

echo "=== P6 riga commentata: non deve dare falso verde ==="
mkdir -p $T/h6
printf '# export PATH="$HOME/.mtplx/bin:$PATH"\n' > $T/h6/.zshrc
out=$(runf $T/h6 "" "" --check); rc=$?
{ [ $rc = 1 ] && echo "$out" | grep -q 'PATH: no'; } && ok "commentata != presente" || no "falso verde :: $out"
out=$(run $T/h6)
[ "$(grep -c '^export PATH=' $T/h6/.zshrc)" = 1 ] && ok "aggiunta UNA riga vera, il commento resta" || no "conteggio $(grep -c '^export PATH=' $T/h6/.zshrc)"

echo "=== P7 PATH mancante in --check: drift, zero scritture ==="
mkdir -p $T/h7; printf 'alias ls="ls -G"\n' > $T/h7/.zshrc
Z=$(shasum $T/h7/.zshrc|cut -d' ' -f1)
out=$(runf $T/h7 "" "" --check); rc=$?
{ [ $rc = 1 ] && echo "$out" | grep -q 'PATH: no'; } && ok "exit 1 + diagnosi" || no "rc=$rc :: $out"
[ "$Z" = "$(shasum $T/h7/.zshrc|cut -d' ' -f1)" ] && ok "rc intatta in --check" || no "scritto"

printf "%s
" "PASS=$P FAIL=$F" > "$RESULT_DIR/P-path.result"
}

suite_B() {
B=$RUNDIR/B; rm -rf $B; mkdir -p $B/home
P=0;F=0; ok(){ echo "  OK   $1"; P=$((P+1)); }; no(){ echo "  FAIL $1"; F=$((F+1)); }
R=$B/fix; R2=$B/fix-moved
export HOME=$B/home MTPLX_HOME=$B/home/.mtplx

echo "=== fixture con uv VERO (python isolato dalla cwd, come fara' lo script) ==="
mkdir -p $R/mtplx $R/scripts
cat > $R/pyproject.toml <<'PP'
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
[project]
name = "mtplx"
version = "0.0.1"
description = "installer fixture"
requires-python = ">=3.11"
dependencies = []
[project.scripts]
mtplx = "mtplx.__main__:main"
[tool.setuptools]
packages = ["mtplx"]
PP
printf 'VERSION = "fixture-1"\n' > $R/mtplx/__init__.py
printf 'from mtplx import VERSION\ndef main():\n    print("mtplx fixture", VERSION)\n' > $R/mtplx/__main__.py
cp "$SRC" $R/scripts/install_repo_launcher.sh
out=$(cd $R && timeout 600 uv sync 2>&1) && ok "uv sync del fixture ok" || { echo "$out"|tail -3; no "sync fallito"; }
res=$(cd / && $R/.venv/bin/python3 -P -c 'import mtplx; print(mtplx.__file__)' 2>&1)
echo "    import (da cwd=/): $res"
[ -f "$res" ] && ok "editable risolve nel fixture, senza aiuto della cwd" || no "non risolve"
before=$( (cd / && $R/.venv/bin/python3 -P -c 'import mtplx,os; print(os.path.realpath(mtplx.__file__))') 2>&1 )
echo "=== 1. install normale ==="
out=$(bash $R/scripts/install_repo_launcher.sh 2>&1); rc=$?
echo "$out" | grep -q 'venv: serves' && ok "riconosciuto sano" || no "rc=$rc :: $out"
$B/home/.mtplx/bin/mtplx 2>&1 | grep -q 'fixture-1' && ok "il launcher esegue il fixture" || no "esecuzione rotta"

echo "=== 2. sposto il repo (nessun --force, nessuna mano mia sui path) ==="
mv $R $R; mv $B/fix $R2; R=$R2
echo "    import dopo lo spostamento: $( (cd / && $R/.venv/bin/python3 -P -c 'import mtplx; print(mtplx.__file__)') 2>&1 )"
echo "    shebang console script: $(head -1 $R/.venv/bin/mtplx)"
chk=$(bash $R/scripts/install_repo_launcher.sh --check 2>&1); rc=$?
[ $rc = 1 ] && ok "--check => exit 1" || no "rc=$rc :: $chk"
echo "$chk" | grep -q 'needs uv sync' && ok "--check becca il VENV rotto (il bug di prima)" || no "venv non rilevato :: $chk"
echo "$chk" | grep -q 'stale, generated for' && ok "--check becca il LAUNCHER stale" || no "launcher non rilevato"

echo "=== 3. riparazione automatica, uv sync reale ==="
out=$(bash $R/scripts/install_repo_launcher.sh 2>&1); rc=$?
echo "$out" | grep -q 'running: (cd' && ok "esegue uv sync da solo" || no "non eseguito :: $out"
echo "$out" | grep -q 'venv: repaired' && ok "riverifica e conferma" || no "conferma mancante :: $out"
[ $rc = 0 ] && ok "exit 0" || no "rc=$rc :: $out"
res=$( (cd / && $R/.venv/bin/python3 -P -c 'import mtplx,os; print(os.path.realpath(mtplx.__file__))') 2>&1 )
want=$( (cd $R && pwd -P) )
echo "    import dopo sync: $res"
[ "$(dirname "$res")" = "$want/mtplx" ] && ok "mtplx risolve nella NUOVA posizione" || no "risolve ancora altrove"
$B/home/.mtplx/bin/mtplx 2>&1 | grep -q 'fixture-1' && ok "il launcher aggiornato ESEGUIVE il fixture spostato" || no "ancora rotto"
head -1 $R/.venv/bin/mtplx | grep -q "$want" && ok "shebang rigenerato da uv sul nuovo path" || no "shebang vecchio: $(head -1 $R/.venv/bin/mtplx)"
grep -q "^# repo: " $B/home/.mtplx/bin/mtplx && ok "provenienza presente" || no "provenienza persa"
# funzionale = cio' che viene eseguito: console script, finder, launcher
if grep -q "irlB2/fix/" $R/.venv/bin/mtplx 2>/dev/null; then no "console script ancora sul path vecchio"; else ok "console script pulito dal path vecchio"; fi
if grep -rq "irlB2/fix/" $R/.venv/lib/python*/site-packages/__editable__* 2>/dev/null; then
  no "finder editable ancora sul path vecchio"
else
  ok "finder editable pulito dal path vecchio"
fi
# limite noto, documentato: uv sync NON riscrive gli activate.*, che restano sul path di creazione
if grep -q "irlB2/fix/.venv" $R/.venv/bin/activate 2>/dev/null; then
  ok "limite noto confermato: activate. tiene il path vecchio (non usato dal launcher)"
else
  echo "    (nota: activate riscritto in questo giro, il limite non si e' presentato)"
fi

echo "=== 4. stabilita' ==="
bash $R/scripts/install_repo_launcher.sh --check >/dev/null 2>&1; [ $? = 0 ] && ok "--check pulito dopo la riparazione" || no "drift residuo"
out=$(bash $R/scripts/install_repo_launcher.sh 2>&1)
echo "$out" | grep -q 'already installed for this checkout' && ok "secondo giro: nessuna riscrittura" || no "ribatte :: $out"
printf "%s
" "PASS=$P FAIL=$F" > "$RESULT_DIR/B-real-uv.result"
rm -rf $B
}

run_suite() {
  local name=$1 fn=$2
  printf '\n########## %s ##########\n' "$name"
  ( $fn )
  if [ ! -f "$RESULT_DIR/$name.result" ]; then
    printf '  FAIL %s produced no result file\n' "$name"
    echo "PASS=0 FAIL=1" > "$RESULT_DIR/$name.result"
  fi
}

run_suite A-mechanism suite_A
run_suite P-path suite_P
if [ "$with_real" = 1 ]; then
  if command -v uv >/dev/null 2>&1; then
    run_suite B-real-uv suite_B
  else
    printf '\n  SKIP B-real-uv: `uv` not on PATH\n'
    echo "PASS=0 FAIL=0" > "$RESULT_DIR/B-real-uv.result"
  fi
else
  printf '\n  B-real-uv non eseguito (passa --all: usa uv reale e scarica il build backend)\n'
fi

printf '\n########## RIEPILOGO ##########\n'
tp=0; tf=0; bad=0
for f in "$RESULT_DIR"/*.result; do
  [ -e "$f" ] || continue
  line=$(cat "$f"); n=$(basename "$f" .result)
  p=${line#PASS=}; p=${p%% *}; fq=${line##*FAIL=}
  printf '  %-14s PASS=%s FAIL=%s\n' "$n" "$p" "$fq"
  tp=$((tp+p)); tf=$((tf+fq)); [ "$fq" = 0 ] || bad=1
done
printf '  %-14s PASS=%s FAIL=%s\n' TOTALE "$tp" "$tf"
if [ "$bad" = 0 ]; then echo "tutto verde"; exit 0; else echo "fallimenti sopra"; exit 1; fi
