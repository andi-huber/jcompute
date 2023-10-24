/*
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */
package jcompute.core.mem.buffered;

import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;

import jcompute.core.mem.ByteMemory;
import jcompute.core.shape.Shape;
import jcompute.core.util.function.LongToByteFunction;
import jcompute.core.util.io.ChannelUtils;
import lombok.SneakyThrows;

public record ByteMemoryBuffered(Shape shape, ByteBuffer buffer)
implements ByteMemory<ByteMemoryBuffered> {

    public ByteMemoryBuffered(final Shape shape) {
        this(shape, ByteBuffer.allocate(Math.toIntExact(shape.totalSize())));
    }

    @Override
    public ByteMemoryBuffered put(final long gid, final byte value) {
        buffer.put(Math.toIntExact(gid), value);
        return this;
    }

    @Override
    public ByteMemoryBuffered fill(final LongToByteFunction filler) {
        shape().forEach(gid->put(gid, filler.applyAsByte(gid)));
        return this;
    }

    @Override
    public byte get(final long gid) {
        return buffer.get(Math.toIntExact(gid));
    }

    @SneakyThrows
    public void write(final OutputStream out) {
        shape.write(out);
        ChannelUtils.write(buffer, out);
    }

    @Override
    public boolean equals(final Object obj) {
        return ByteMemory.equals(this, obj);
    }

    // -- FACTORIES

    @SneakyThrows
    public static ByteMemoryBuffered read(final InputStream in) {
        var shape = Shape.read(in);
        final int size = Math.toIntExact(shape.totalSize());
        return new ByteMemoryBuffered(shape, ChannelUtils.readBytes(size, in));
    }

}