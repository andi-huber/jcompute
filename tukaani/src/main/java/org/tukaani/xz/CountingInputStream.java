/*
 * CountingInputStream
 *
 * Author: Lasse Collin <lasse.collin@tukaani.org>
 *
 * This file has been put into the public domain.
 * You can do whatever you want with this file.
 */

package org.tukaani.xz;

import java.io.IOException;
import java.io.InputStream;

/**
 * Counts the number of bytes read from an input stream.
 * The <code>close()</code> method does nothing, that is, the underlying
 * <code>InputStream</code> isn't closed.
 */
class CountingInputStream extends CloseIgnoringInputStream {
    private long size = 0;

    public CountingInputStream(final InputStream in) {
        super(in);
    }

    @Override
    public int read() throws IOException {
        int ret = in.read();
        if (ret != -1 && size >= 0)
            ++size;

        return ret;
    }

    @Override
    public int read(final byte[] b, final int off, final int len) throws IOException {
        int ret = in.read(b, off, len);
        if (ret > 0 && size >= 0)
            size += ret;

        return ret;
    }

    public long getSize() {
        return size;
    }
}
