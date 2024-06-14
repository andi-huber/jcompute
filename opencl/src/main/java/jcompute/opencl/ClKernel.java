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

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_kernel;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.val;
import lombok.experimental.Accessors;

@RequiredArgsConstructor
public class ClKernel implements ClResource {

    @Getter @Accessors(fluent = true) private final cl_kernel id;
    @Getter private final ClProgram program;
    @Getter private final String name;

    /**
     * Add OpenCL kernel argument.
     */
    public ClKernel setArg(final int argIndex, final ClMem memObj) {
        return setArg(this, argIndex, Sizeof.cl_mem, Pointer.to(memObj.id()));
    }

    public ClKernel setArg(final int argIndex, final byte value) {
        return setArg(this, argIndex, 1, Pointer.to(new byte[]{ value }));
    }

    public ClKernel setArg(final int argIndex, final short value) {
        return setArg(this, argIndex, Sizeof.cl_short, Pointer.to(new short[]{ value }));
    }

    public ClKernel setArg(final int argIndex, final int value) {
        return setArg(this, argIndex, Sizeof.cl_int, Pointer.to(new int[]{ value }));
    }

    public ClKernel setArg(final int argIndex, final long value) {
        return setArg(this, argIndex, Sizeof.cl_long, Pointer.to(new long[]{ value }));
    }

    public ClKernel setArg(final int argIndex, final float value) {
        return setArg(this, argIndex, Sizeof.cl_float, Pointer.to(new float[]{ value }));
    }

    public ClKernel setArg(final int argIndex, final double value) {
        return setArg(this, argIndex, Sizeof.cl_double, Pointer.to(new double[]{ value }));
    }

    @Override
    public void free() {
        final int ret = CL.clReleaseKernel(id());
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
        val ret_pointer = new int[1];
        val kernelId = CL.clCreateKernel(program.id(), kernelName, ret_pointer);
        int ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
                String.format("failed to create kernel '%s' for program %s", kernelName, program));
        return new ClKernel(kernelId, program, kernelName);
    }

    /* Set OpenCL kernel argument */
    private ClKernel setArg(final ClKernel kernel, final int argIndex, final long sizeOf, final Pointer pointer) {
        int ret = CL.clSetKernelArg(kernel.id(), argIndex, sizeOf, pointer);
        _Util.assertSuccess(ret, ()->
            String.format("failed to set kernel argument for kernel %s", kernel.getName()));
        return kernel;
    }

}
