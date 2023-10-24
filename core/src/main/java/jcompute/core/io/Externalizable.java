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
package jcompute.core.io;

import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;

import lombok.SneakyThrows;

public interface Externalizable<T> {

    void writeExternal(ObjectOutput out) throws IOException;
    T readExternal(ObjectInput in) throws IOException, ClassNotFoundException;

    @SneakyThrows
    default T write(final OutputStream os, final Compressor compressor) {
        try(var oos = new ObjectOutputStream(compressor.out(os))){
            writeExternal(oos);
        }
        return (T)this;
    }

    @SneakyThrows
    default T read(final InputStream is, final Compressor compressor) {
        try(var ois = new ObjectInputStream(compressor.in(is))){
            return readExternal(ois);
        }
    }

}
