const { execSync } = require("child_process");

function run(cmd) {
  try {
    execSync(cmd, { stdio: "ignore", shell: "cmd.exe" });
  } catch {}
}

run("taskkill /F /IM uvicorn.exe 2>nul");

for (const port of [8000, 3000]) {
  run(
    `for /f "tokens=5" %a in ('netstat -ano ^| findstr :${port} ^| findstr LISTENING') do taskkill /F /PID %a`
  );
}

console.log("Killed stale backend/frontend processes (ports 8000, 3000)");
