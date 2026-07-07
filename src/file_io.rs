use std::fs::File;

pub(crate) fn read_exact_at(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    read_exact_at_impl(file, offset, buf)
}

#[cfg(unix)]
fn read_exact_at_impl(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    use std::os::unix::fs::FileExt;

    let mut read = 0;
    while read < buf.len() {
        let n = file.read_at(&mut buf[read..], offset + read as u64)?;
        if n == 0 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "failed to fill buffer",
            ));
        }
        read += n;
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at_impl(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    use std::os::windows::fs::FileExt;

    let mut read = 0;
    while read < buf.len() {
        let n = file.seek_read(&mut buf[read..], offset + read as u64)?;
        if n == 0 {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "failed to fill buffer",
            ));
        }
        read += n;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at_impl(file: &mut File, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::{Read, Seek, SeekFrom};

    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}
