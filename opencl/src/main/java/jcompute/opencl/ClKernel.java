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

import static org.bytedeco.opencl.global.OpenCL.clCreateKernel;
import static org.bytedeco.opencl.global.OpenCL.clSetKernelArg;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.val;
import lombok.experimental.Accessors;

@RequiredArgsConstructor
public class ClKernel implements ClResource {

    @Getter @Accessors(fluent = true) private final _cl_kernel id;
    @Getter private final ClProgram program;
    @Getter private final String name;

    /**
     * Add OpenCL kernel argument.
     */
    public ClKernel setArg(final int argIndex, final ClMem memObj) {
        try(val pointer = new PointerPointer<>(1)){
            return setArg(this, argIndex, pointer.put(memObj.id()));
        }
    }

    public ClKernel setArg(final int argIndex, final byte value) {
        try(val pointer = new BytePointer(1)){
            return setArg(this, argIndex, pointer.put(value));
        }
    }

    public ClKernel setArg(final int argIndex, final short value) {
        try(val pointer = new ShortPointer(1)){
            return setArg(this, argIndex, pointer.put(value));
        }
    }

    public ClKernel setArg(final int argIndex, final int value) {
        try(val pointer = new IntPointer(1)){
            return setArg(this, argIndex, pointer.put(value));
        }
    }

    public ClKernel setArg(final int argIndex, final long value) {
        try(val pointer = new LongPointer(1)){
            return setArg(this, argIndex, pointer.put(value));
        }
    }

    public ClKernel setArg(final int argIndex, final float value) {
        try(val pointer = new FloatPointer(1)){
            return setArg(this, argIndex, pointer.put(value));
        }
    }

    public ClKernel setArg(final int argIndex, final double value) {
        try(val pointer = new DoublePointer(1)){
            return setArg(this, argIndex, pointer.put(value));
        }
    }

    @Override
    public void free() {
        final int ret = OpenCL.clReleaseKernel(id());
        _Util.assertSuccess(ret, ()->
            String.format("failed to release kernel '%s' for program %s", name, program));
    }

    // -- UTILITY

    public ClKernel setArgs(final Object... args) {
        for (int i = 0; i < args.length; i++) {
            val value = args[i];
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

    // -- HELPER

    /**
     * Returns a new OpenCL kernel for given program.
     */
    static ClKernel createKernel(final ClProgram program, final String kernelName) {
        val ret_pointer = new IntPointer(1);
        val kernelId = clCreateKernel(program.id(), kernelName, ret_pointer);
        val ret = ret_pointer.get();
        _Util.assertSuccess(ret, ()->
                String.format("failed to create kernel '%s' for program %s", kernelName, program));
        return new ClKernel(kernelId, program, kernelName);
    }

    /* Set OpenCL kernel argument */
    private ClKernel setArg(final ClKernel kernel, final int argIndex, final Pointer pointer) {
        val ret = clSetKernelArg(kernel.id(),
                argIndex,
                Loader.sizeof(pointer.getClass()),
                pointer);
        _Util.assertSuccess(ret, ()->
            String.format("failed to set kernel argument for kernel %s", kernel.getName()));
        return kernel;
    }

}
