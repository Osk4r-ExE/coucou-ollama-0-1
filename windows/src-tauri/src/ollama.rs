// Ollama API client for local LLM inference
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};
use std::sync::{Mutex, Arc};
use std::thread;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use once_cell::sync::Lazy;

const OLLAMA_HOST: &str = "http://localhost:11434";
const IDLE_TIMEOUT: Duration = Duration::from_secs(1800);
const SYSTEM_PROMPT: &str = "You are Mochi, a personal AI assistant. Respond in the user's language. Be thorough. No markdown formatting. Use plain text.";

pub struct OllamaManager {
    process: Mutex<Option<std::process::Child>>,
    last_activity: Mutex<Instant>,
    is_running: Mutex<bool>,
}

impl OllamaManager {
    pub fn new() -> Self {
        Self {
            process: Mutex::new(None),
            last_activity: Mutex::new(Instant::now()),
            is_running: Mutex::new(false),
        }
    }
    pub fn start(&self) -> Result<(), String> {
        let mut p = self.process.lock().unwrap();
        let mut r = self.is_running.lock().unwrap();
        if *r { return Ok(()); }
        let Ok(child) = Command::new("ollama").arg("serve").stdout(Stdio::null()).stderr(Stdio::null()).spawn() else {
            return Err("Failed to start ollama".into());
        };
        *p = Some(child);
        *r = true;
        *self.last_activity.lock().unwrap() = Instant::now();
        std::thread::sleep(Duration::from_secs(2));
        Ok(())
    }
    pub fn stop(&self) -> Result<(), String> {
        let mut p = self.process.lock().unwrap();
        let mut r = self.is_running.lock().unwrap();
        if let Some(mut child) = p.take() {
            let _ = child.kill();
            std::thread::sleep(Duration::from_millis(500));
            *r = false;
        }
        Ok(())
    }
    pub fn is_running(&self) -> bool { *self.is_running.lock().unwrap() }
    pub fn update_activity(&self) { *self.last_activity.lock().unwrap() = Instant::now(); }
    pub fn check_idle(&self) -> bool { self.last_activity.lock().unwrap().elapsed() >= IDLE_TIMEOUT }
    pub async fn get_models() -> Result<Vec<String>, String> {
        let client = reqwest::Client::new();
        let resp = client.get(&format!("{}/api/tags", OLLAMA_HOST)).timeout(Duration::from_secs(5)).send().await.map_err(|e| e.to_string())?;
        let json: Value = resp.json().await.map_err(|e| e.to_string())?;
        let models = json["models"].as_array().ok_or("No models")?;
        let names: Vec<String> = models.iter().filter_map(|m| m["name"].as_str()).map(String::from).collect();
        Ok(names)
    }
    pub async fn check_health() -> bool {
        let client = reqwest::Client::new();
        match client.get(&format!("{}/api/tags", OLLAMA_HOST)).timeout(Duration::from_secs(5)).send().await {
            Ok(r) => r.status().is_success(), Err(_) => false
        }
    }
}

pub static OLLAMA_MANAGER: Lazy<Arc<OllamaManager>> = Lazy::new(|| Arc::new(OllamaManager::new()));

pub fn start_idle_monitor() {
    std::thread::spawn(|| loop {
        std::thread::sleep(Duration::from_secs(60));
        if OLLAMA_MANAGER.is_running() && OLLAMA_MANAGER.check_idle() {
            let _ = OLLAMA_MANAGER.stop();
        }
    });
}

#[derive(Serialize)]
pub struct OllamaReply { pub text: String }

#[derive(Debug, Clone, Deserialize)]
pub struct OllamaChatContext { pub kind: String, pub name: Option<String>, pub path: Option<String> }

pub async fn ollama_chat_send(message: &str, model: &str, _history: Vec<Value>, _context: Option<OllamaChatContext>) -> Result<String, String> {
    OLLAMA_MANAGER.update_activity();
    if !OLLAMA_MANAGER.is_running() {
        OLLAMA_MANAGER.start()?;
        std::thread::sleep(Duration::from_secs(2));
    }
    let client = reqwest::Client::new();
    let body = json!({"model": model, "stream": false, "messages": [{"role": "user", "content": message}]});
    let resp = client.post(&format!("{}/api/chat", OLLAMA_HOST)).json(&body).send().await.map_err(|e| e.to_string())?;
    let json: Value = resp.json().await.map_err(|e| e.to_string())?;
    let text = json["message"]["content"].as_str().ok_or("No text")?.to_string();
    Ok(text)
}
