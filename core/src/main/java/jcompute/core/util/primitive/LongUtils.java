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

import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;

import lombok.SneakyThrows;
import lombok.experimental.UtilityClass;

@UtilityClass
public class LongUtils {

    @Deprecated
    public static record LongHelper(ByteBuffer byteBuffer) {
        public LongHelper() {
            this(ByteBuffer.allocate(Long.BYTES));
        }
        public long fromBytes(final byte[] bytes) {
            byteBuffer.put(0, bytes);
            byteBuffer.rewind();
            long ref = byteBuffer.getLong();
            long x = LongUtils.fromBytes(bytes);

            if(ref!=x) {
                System.err.printf("ref %s%n", Long.toBinaryString(ref));
                System.err.printf("x   %s%n", Long.toBinaryString(x));
                System.exit(1);
            }

            return ref;
        }
//        @SneakyThrows
//        public int write(final long value, final WritableByteChannel channel) {
//            byteBuffer.putLong(0, value);
//            byteBuffer.rewind();
//            return channel.write(byteBuffer);
//        }
//        @SneakyThrows
//        public long read(final ReadableByteChannel channel) {
//            byteBuffer.rewind();
//            if(channel.read(byteBuffer)!=Long.BYTES) {
//                throw new IOException(String.format("Could not read 8 bytes to assemble a long."));
//            }
//            byteBuffer.rewind();
//            return byteBuffer.getLong();
//        }
        @SneakyThrows
        public long read(final InputStream in) {
            return fromBytes(in.readNBytes(Long.BYTES));
        }
    }

    public void toBytes(final long data, final byte[] bytes) {
        bytes[0] = (byte)(data >> 56);
        bytes[1] = (byte)(data >> 48);
        bytes[2] = (byte)(data >> 40);
        bytes[3] = (byte)(data >> 32);
        bytes[4] = (byte)(data >> 24);
        bytes[5] = (byte)(data >> 16);
        bytes[6] = (byte)(data >> 8);
        bytes[7] = (byte)(data);
    }

    public void toBytes(final int limit, final long[] data, final byte[] bytes) {
        int j = 0;
        for (int i = 0; i < limit; i++) {
            final long v = data[i];
            bytes[j++] = (byte)(v >> 56);
            bytes[j++] = (byte)(v >> 48);
            bytes[j++] = (byte)(v >> 40);
            bytes[j++] = (byte)(v >> 32);
            bytes[j++] = (byte)(v >> 24);
            bytes[j++] = (byte)(v >> 16);
            bytes[j++] = (byte)(v >> 8);
            bytes[j++] = (byte)(v);
        }
    }

    public long fromBytes(final byte[] bytes) {
        long v = 0L;
        v|= (bytes[0] & 0xffL) << 56;
        v|= (bytes[1] & 0xffL) << 48;
        v|= (bytes[2] & 0xffL) << 40;
        v|= (bytes[3] & 0xffL) << 32;
        v|= (bytes[4] & 0xffL) << 24;
        v|= (bytes[5] & 0xffL) << 16;
        v|= (bytes[6] & 0xffL) << 8;
        v|= (bytes[7] & 0xffL) << 0;
        return v;
    }

    public long pack(final int mostSignificant, final int leastSignificant) {
        return ((mostSignificant & 0xffff_ffffL)<<32)
                | (leastSignificant & 0xffff_ffffL);
    }

    @SneakyThrows
    public void write(final long value, final OutputStream out) {
        var byteArray = new byte[Long.BYTES];
        toBytes(value, byteArray);
        out.write(byteArray);
    }

    @SneakyThrows
    public void writeBuffer(final LongBuffer buffer, final OutputStream out) {
        var n = 64;
        var longArray = new long[n];
        var byteArray = new byte[Long.BYTES * n];
        buffer.rewind();
        int remainingLongs = buffer.capacity();
        while(remainingLongs>0) {
            int size = Math.min(remainingLongs, n);
            buffer.get(longArray, 0, size);
            toBytes(size, longArray, byteArray);
            out.write(byteArray, 0, size * Long.BYTES);
            remainingLongs-=size;
        }
    }

    @SneakyThrows
    public LongBuffer readBuffer(final int size, final InputStream in) {
        var buffer = LongBuffer.allocate(size);
        var longHelper = new LongHelper();

        int remainingLongs = size;
        while(remainingLongs>0) {
            buffer.put(longHelper.fromBytes(in.readNBytes(Long.BYTES)));
            remainingLongs-=1;
        }
        return buffer;
    }

    // JUnit
    long[] samples() {
        return new long[] {
                Long.MIN_VALUE,
                -4_123_456_789_012_345_678L,
                -1L,
                0L,
                1L,
                4_123_456_789_012_345_678L,
                Long.MAX_VALUE,
        };
    }

}
