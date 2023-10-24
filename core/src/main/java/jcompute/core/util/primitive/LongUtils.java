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
package jcompute.core.util.primitive;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.ReadableByteChannel;
import java.nio.channels.WritableByteChannel;

import lombok.SneakyThrows;
import lombok.experimental.UtilityClass;

@UtilityClass
public class LongUtils {

    public static record LongHelper(ByteBuffer byteBuffer) {
        public LongHelper() {
            this(ByteBuffer.allocate(Long.BYTES));
        }
        public byte[] toBytes(final long value) {
            byteBuffer.putLong(0, value);
            return byteBuffer.array();
        }
        public long fromBytes(final byte[] bytes) {
            byteBuffer.put(0, bytes);
            byteBuffer.rewind();
            return byteBuffer.getLong();
        }
        @SneakyThrows
        public int write(final long value, final WritableByteChannel channel) {
            byteBuffer.putLong(0, value);
            byteBuffer.rewind();
            return channel.write(byteBuffer);
        }
        @SneakyThrows
        public long read(final ReadableByteChannel channel) {
            byteBuffer.rewind();
            if(channel.read(byteBuffer)!=Long.BYTES) {
                throw new IOException(String.format("Could not read 8 bytes to assemble a long."));
            }
            byteBuffer.rewind();
            return byteBuffer.getLong();
        }
        @SneakyThrows
        public void write(final long value, final OutputStream out) {
            out.write(toBytes(value));
        }
        @SneakyThrows
        public long read(final InputStream in) {
            return fromBytes(in.readNBytes(Long.BYTES));
        }
    }

}
