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
package jcompute.opencl;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

@RequiredArgsConstructor(access = AccessLevel.PROTECTED)
public abstract class ClKernel implements ClResource {

    @Getter @Accessors(fluent = true) private final ClProgram program;
    @Getter @Accessors(fluent = true) private final String name;

    protected abstract int releaseKernel();

    @Override
    public final void free() {
        _Util.assertSuccess(releaseKernel(), ()->
            String.format("failed to release kernel '%s' for program %s", name, program));
    }

    public abstract ClKernel setArg(final int argIndex, final ClMem memObj);

    public abstract ClKernel setArg(final int argIndex, final byte value);
    public abstract ClKernel setArg(final int argIndex, final short value);
    public abstract ClKernel setArg(final int argIndex, final int value);
    public abstract ClKernel setArg(final int argIndex, final long value);
    public abstract ClKernel setArg(final int argIndex, final float value);
    public abstract ClKernel setArg(final int argIndex, final double value);

    // -- UTILITY

    public final ClKernel setArgs(final Object... args) {
        for (int i = 0; i < args.length; i++) {
            var value = args[i];
            if(value instanceof ClMem clMem) {
                setArg(i, clMem);
            } else if(value instanceof Integer v) {
                setArg(i, v);
            } else if(value instanceof Long v) {
                setArg(i, v);
            } else if(value instanceof Float v) {
                setArg(i, v);
            } else if(value instanceof Double v) {
                setArg(i, v);
            } else if(value instanceof Short v) {
                setArg(i, v);
            } else if(value instanceof Byte v) {
                setArg(i, v);
            } else {
                throw new IllegalArgumentException("Unexpected value: " + value.getClass());
            }
        }
        return this;
    }

}
