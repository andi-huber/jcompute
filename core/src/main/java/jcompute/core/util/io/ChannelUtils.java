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
package jcompute.core.util.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.nio.channels.Channels;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;

import jcompute.core.util.primitive.LongUtils.LongHelper;
import lombok.SneakyThrows;
import lombok.experimental.UtilityClass;

@UtilityClass
public class ChannelUtils {

    @SneakyThrows
    public void write(final ByteBuffer buffer, final OutputStream out) {
        final int size = buffer.capacity();
        try(WritableByteChannel channel = Channels.newChannel(out)){
            buffer.rewind();
            int count = channel.write(buffer);
            if(count!=size) {
                throw new IOException(String.format("Could only write %d of %d bytes.", count, size));
            }
        }
    }

    @SneakyThrows
    public static ByteBuffer readBytes(final int size, final InputStream in) {
        var buffer = ByteBuffer.allocate(size);
        buffer.rewind();
        try(ReadableByteChannel channel = Channels.newChannel(in)) {
            int count = 0;
            while(count<size) {
                int read = channel.read(buffer);
                if(read==-1) break;
                count+= read;
            }
            if(count!=size) {
                throw new IOException(String.format("Could only read %d of %d bytes.", count, size));
            }
            return buffer;
        }
    }

    @SneakyThrows
    public void write(final LongBuffer buffer, final OutputStream out) {
        var longHelper = new LongHelper();
        final int size = buffer.capacity();
        try(WritableByteChannel channel = Channels.newChannel(out)){
            buffer.rewind();
            for (int i = 0; i < size; i++) {
                int count = longHelper.write(buffer.get(), channel);
                if(count!=8) {
                    throw new IOException(String.format("Could only write %d of %d longs.", i, size));
                }

            }
        }
    }

    @SneakyThrows
    public static LongBuffer readLongs(final int size, final InputStream in) {
        var longHelper = new LongHelper();
        var buffer = LongBuffer.allocate(size);
        buffer.rewind();
        try(ReadableByteChannel channel = Channels.newChannel(in)) {
            int count = 0;
            while(count<size) {
                buffer.put(longHelper.read(channel));
                count++;
            }
            if(count!=size) {
                throw new IOException(String.format("Could only read %d of %d longs.", count, size));
            }
            return buffer;
        }
    }

}
