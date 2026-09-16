//! Content-addressed UTF-8 blobs in an application-owned, session-isolated directory.
//! The directory and its ancestors must not be writable by untrusted users.
//! This is not a filesystem sandbox. No automatic deletion/retention is performed.
#![forbid(unsafe_code)]

use agentscope::{OffloadStore, OffloadedTextChunk, ToolError, ToolFuture};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
/// A session-scoped local store. Async operations require a Tokio runtime.
pub struct FileOffloadStore {
    root: PathBuf,
}

fn error(message: &str) -> ToolError {
    ToolError::new(message).with_code("offload_storage")
}

impl FileOffloadStore {
    /// Opens/creates a private, per-session directory. Call outside async hot paths.
    /// # Errors
    /// Returns an error if the directory cannot be created or is a symlink.
    pub fn new(root: impl AsRef<Path>) -> Result<Self, ToolError> {
        let root = root.as_ref();
        let mut builder = fs::DirBuilder::new();
        builder.recursive(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(root)
            .map_err(|_| error("cannot create offload directory"))?;
        if !fs::symlink_metadata(root)
            .map_err(|_| error("cannot inspect offload directory"))?
            .is_dir()
        {
            return Err(error("offload root must be a real directory"));
        }
        Ok(Self {
            root: fs::canonicalize(root).map_err(|_| error("cannot resolve offload directory"))?,
        })
    }

    fn path(&self, id: &str) -> Result<PathBuf, ToolError> {
        if id.len() != 64
            || !id
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return Err(error("invalid offload ID"));
        }
        Ok(self.root.join(id))
    }

    fn open(&self, id: &str) -> Result<File, ToolError> {
        let path = self.path(id)?;
        if !fs::symlink_metadata(&path)
            .map_err(|_| error("offloaded text unavailable"))?
            .is_file()
        {
            return Err(error("offloaded text must be a regular file"));
        }
        File::open(path).map_err(|_| error("cannot open offloaded text"))
    }

    fn put_sync(&self, text: &str) -> Result<String, ToolError> {
        let id = format!("{:x}", Sha256::digest(text.as_bytes()));
        let path = self.path(&id)?;
        let mut temp = tempfile::NamedTempFile::new_in(&self.root)
            .map_err(|_| error("cannot create offload file"))?;
        temp.write_all(text.as_bytes())
            .map_err(|_| error("cannot write offloaded text"))?;
        temp.as_file()
            .sync_all()
            .map_err(|_| error("cannot sync offloaded text"))?;
        match temp.persist_noclobber(&path) {
            Ok(_) => {}
            Err(e) if e.error.kind() == std::io::ErrorKind::AlreadyExists => {
                let mut file = self.open(&id)?;
                if file
                    .metadata()
                    .map_err(|_| error("cannot inspect offloaded text"))?
                    .len()
                    != text.len() as u64
                {
                    return Err(error("existing offload content is corrupt"));
                }
                let mut hash = Sha256::new();
                let mut buffer = [0u8; 8192];
                loop {
                    let count = file
                        .read(&mut buffer)
                        .map_err(|_| error("cannot verify offloaded text"))?;
                    if count == 0 {
                        break;
                    }
                    hash.update(&buffer[..count]);
                }
                if format!("{:x}", hash.finalize()) != id {
                    return Err(error("existing offload content is corrupt"));
                }
            }
            Err(_) => return Err(error("cannot publish offloaded text")),
        }
        File::open(&self.root)
            .and_then(|directory| directory.sync_all())
            .map_err(|_| error("cannot sync offload directory"))?;
        Ok(id)
    }

    fn read_sync(
        &self,
        id: &str,
        offset: usize,
        max_bytes: usize,
    ) -> Result<OffloadedTextChunk, ToolError> {
        if max_bytes < 4 {
            return Err(error("read limit must be at least four bytes"));
        }
        let mut file = self.open(id)?;
        let total = usize::try_from(
            file.metadata()
                .map_err(|_| error("cannot inspect offloaded text"))?
                .len(),
        )
        .map_err(|_| error("offloaded text too large"))?;
        if offset > total {
            return Err(error("offset exceeds text length"));
        }
        file.seek(SeekFrom::Start(offset as u64))
            .map_err(|_| error("cannot seek offloaded text"))?;
        let mut bytes = vec![0; max_bytes.min(total - offset)];
        file.read_exact(&mut bytes)
            .map_err(|_| error("cannot read offloaded text"))?;
        match std::str::from_utf8(&bytes) {
            Ok(_) => {}
            Err(e) if e.error_len().is_none() && offset + bytes.len() < total => {
                bytes.truncate(e.valid_up_to());
            }
            Err(_) => return Err(error("invalid UTF-8 or offset is not a character boundary")),
        }
        let text = String::from_utf8(bytes).map_err(|_| error("invalid offloaded UTF-8"))?;
        Ok(OffloadedTextChunk {
            next_offset: offset + text.len(),
            total_bytes: total,
            text,
        })
    }
}

impl OffloadStore for FileOffloadStore {
    fn put<'a>(&'a self, text: &'a str) -> ToolFuture<'a, String> {
        let store = self.clone();
        let text = text.to_owned();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || store.put_sync(&text))
                .await
                .map_err(|_| error("offload write task failed"))?
        })
    }
    fn read<'a>(
        &'a self,
        id: &'a str,
        offset: usize,
        max_bytes: usize,
    ) -> ToolFuture<'a, OffloadedTextChunk> {
        let store = self.clone();
        let id = id.to_owned();
        Box::pin(async move {
            tokio::task::spawn_blocking(move || store.read_sync(&id, offset, max_bytes))
                .await
                .map_err(|_| error("offload read task failed"))?
        })
    }
}
