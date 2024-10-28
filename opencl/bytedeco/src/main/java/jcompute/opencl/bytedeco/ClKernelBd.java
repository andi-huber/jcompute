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
package jcompute.opencl.bytedeco;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.ShortPointer;
import org.bytedeco.opencl._cl_kernel;
import org.bytedeco.opencl.global.OpenCL;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.opencl.ClKernel;
import jcompute.opencl.ClMem;
import jcompute.opencl.ClProgram;

public final class ClKernelBd extends ClKernel {

    @Getter @Accessors(fluent = true) private final _cl_kernel id;

    ClKernelBd(final _cl_kernel id, final ClProgram program, final String name) {
        super(program, name);
        this.id = id;
    }

    @Override
    protected int releaseKernel() {
        return OpenCL.clReleaseKernel(id());
    }

    /**
     * Add OpenCL kernel argument.
     */
    @Override
    public ClKernel setArg(final int argIndex, final ClMem memObj) {
        try(var pointer = new PointerPointer<>(1)){
            var sizeOf = Loader.sizeof(pointer.getClass());
            return setArg(this, argIndex, sizeOf, pointer.put(((ClMemBd)memObj).id()));
        }
    }

    @Override
    public ClKernel setArg(final int argIndex, final byte value) {
        try(var pointer = new BytePointer(1)){
            return setArg(this, argIndex, 1, pointer.put(value));
        }
    }

    @Override
    public ClKernel setArg(final int argIndex, final short value) {
        try(var pointer = new ShortPointer(1)){
            return setArg(this, argIndex, 2, pointer.put(value));
        }
    }

    @Override
    public ClKernel setArg(final int argIndex, final int value) {
        try(var pointer = new IntPointer(1)){
            return setArg(this, argIndex, 4, pointer.put(value));
        }
    }

    @Override
    public ClKernel setArg(final int argIndex, final long value) {
        try(var pointer = new LongPointer(1)){
            return setArg(this, argIndex, 8, pointer.put(value));
        }
    }

    @Override
    public ClKernel setArg(final int argIndex, final float value) {
        try(var pointer = new FloatPointer(1)){
            return setArg(this, argIndex, 4, pointer.put(value));
        }
    }

    @Override
    public ClKernel setArg(final int argIndex, final double value) {
        try(var pointer = new DoublePointer(1)){
            return setArg(this, argIndex, 8, pointer.put(value));
        }
    }

    /* Set OpenCL kernel argument */
    private ClKernel setArg(final ClKernelBd kernel, final int argIndex, final long sizeOf, final Pointer pointer) {
        int ret = OpenCL.clSetKernelArg(kernel.id(), argIndex, sizeOf, pointer);
        _Util.assertSuccess(ret, ()->
            String.format("failed to set kernel argument for kernel %s", kernel.name()));
        return kernel;
    }

}
