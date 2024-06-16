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

import lombok.Getter;
import lombok.RequiredArgsConstructor;

import jcompute.core.shape.Shape;

@RequiredArgsConstructor
public abstract class ClCommandQueue implements ClResource {

    @Getter private final ClContext context;

    protected abstract int flushQueue();
    protected abstract int finishQueue();
    protected abstract int releaseQueue();

    @Override
    public final void free() {
        flush();
        finish();
        releaseQueue();
    }

    public abstract  ClCommandQueue enqueueWriteBuffer(
            final ClMem memObj);

    public abstract ClCommandQueue enqueueReadBuffer(
            final ClMem memObj);

    /**
     * localSize is auto
     */
    public abstract ClCommandQueue enqueueNDRangeKernel(
            final ClKernel kernel,
            final Shape globalSize);

    public abstract ClCommandQueue enqueueNDRangeKernel(
            final ClKernel kernel,
            final Shape globalSize,
            final Shape localSize);

    public final ClCommandQueue flush() {
        _Util.assertSuccess(
                flushQueue(), ()->
                    String.format("failed to flush command queue for context %s", context));
        return this;
    }

    public final ClCommandQueue finish() {
        _Util.assertSuccess(
                finishQueue(), ()->
                    String.format("failed to finish command queue for context %s", context));
        return this;
    }

}
