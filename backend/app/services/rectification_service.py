from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any
from app.services.storage import LocalStorage


class RectificationService:
    def __init__(self, storage: LocalStorage, pipeline: Any) -> None:
        self.storage = storage
        self.pipeline = pipeline

    def apply_code_fix(self, repo_id: str, file_path: str, original_code: str, replacement_code: str) -> dict[str, Any]:
        repo_root = self.storage.repo_source_dir(repo_id)
        if not repo_root.exists():
            return {"status": "failed", "error": "Repository source folder not found."}
            
        abs_path = (repo_root / file_path).resolve()
        
        # Security check: Prevent directory traversal out of the repository root
        try:
            if not abs_path.is_relative_to(repo_root.resolve()):
                return {"status": "failed", "error": "Invalid file path: path must reside inside repository root."}
        except ValueError:
            return {"status": "failed", "error": "Invalid file path structure."}

        if not abs_path.exists():
            return {"status": "failed", "error": f"File '{file_path}' does not exist on disk."}

        try:
            # Read existing file content
            content = abs_path.read_text(encoding="utf-8", errors="replace")
            
            # Match validation layers
            target_str = original_code
            new_content = None
            
            # Layer A: Exact match
            if target_str in content:
                new_content = content.replace(target_str, replacement_code, 1)
            else:
                # Layer B: Try matching with stripped leading/trailing whitespaces (common LLM artifact)
                stripped_target = target_str.strip()
                if stripped_target in content:
                    new_content = content.replace(stripped_target, replacement_code.strip(), 1)
                else:
                    # Layer C: Find target block ignoring trailing whitespaces on each line
                    target_lines = [l.rstrip() for l in target_str.splitlines() if l.strip()]
                    content_lines = content.splitlines()
                    
                    match_idx = -1
                    # Basic rolling window search for the block
                    for i in range(len(content_lines) - len(target_lines) + 1):
                        window = [content_lines[i + j].rstrip() for j in range(len(target_lines))]
                        if window == target_lines:
                            match_idx = i
                            break
                            
                    if match_idx != -1:
                        # Reconstruct the file with the replacement
                        before = "\n".join(content_lines[:match_idx])
                        after = "\n".join(content_lines[match_idx + len(target_lines):])
                        new_content = (before + "\n" if before else "") + replacement_code + ("\n" + after if after else "")
            
            if new_content is None:
                return {
                    "status": "failed", 
                    "error": (
                        "Original code block could not be located in the file. "
                        "This can happen if the block was modified previously or formatted differently."
                    )
                }
                
            # Create safety backup file
            backup_path = abs_path.with_suffix(abs_path.suffix + ".bak")
            shutil.copy2(abs_path, backup_path)
            
            # Save updated file
            abs_path.write_text(new_content, encoding="utf-8")
            
            # Re-run pipeline analysis to dynamically rebuild CodeGraph, Graphify and chunks instantly!
            metadata = self.storage.load_repo_metadata(repo_id)
            if metadata:
                self.pipeline.analyze_existing(
                    name=metadata.name, 
                    source_dir=repo_root, 
                    origin=metadata.origin, 
                    repo_id=repo_id
                )
                
            return {
                "status": "success",
                "file_path": file_path,
                "backup_path": str(backup_path.name),
                "message": f"Successfully applied changes to '{file_path}'. A backup copy was created."
            }
            
        except Exception as exc:
            return {"status": "failed", "error": f"Error applying fix: {exc}"}
