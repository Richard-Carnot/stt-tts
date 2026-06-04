import asyncio
import json
import unittest

import vexyl_tts_server as server


class StreamingContractTests(unittest.TestCase):
    def test_cache_key_includes_description_sample_rate_and_generation_params(self):
        base = server.make_cache_key(
            "hello",
            "en-IN",
            "default",
            description="Mary speaks clearly.",
            output_sample_rate=8000,
            generation_params={"do_sample": True, "play_steps": 40},
        )

        self.assertNotEqual(
            base,
            server.make_cache_key(
                "hello",
                "en-IN",
                "default",
                description="Thoma speaks clearly.",
                output_sample_rate=8000,
                generation_params={"do_sample": True, "play_steps": 40},
            ),
        )
        self.assertNotEqual(
            base,
            server.make_cache_key(
                "hello",
                "en-IN",
                "default",
                description="Mary speaks clearly.",
                output_sample_rate=44100,
                generation_params={"do_sample": True, "play_steps": 40},
            ),
        )
        self.assertNotEqual(
            base,
            server.make_cache_key(
                "hello",
                "en-IN",
                "default",
                description="Mary speaks clearly.",
                output_sample_rate=8000,
                generation_params={"do_sample": True, "play_steps": 80},
            ),
        )

    def test_audio_end_omits_full_audio_by_default(self):
        msg = server.build_audio_end_message(
            request_id="req_1",
            total_chunks=2,
            first_chunk_ms=650,
            latency_ms=2400,
            sample_rate=44100,
            full_audio_b64="abc",
            include_full_audio=False,
        )

        self.assertEqual(msg["type"], "audio_end")
        self.assertEqual(msg["first_chunk_ms"], 650)
        self.assertNotIn("full_audio_b64", msg)

    def test_audio_end_can_include_full_audio_for_compatibility(self):
        msg = server.build_audio_end_message(
            request_id="req_1",
            total_chunks=2,
            first_chunk_ms=650,
            latency_ms=2400,
            sample_rate=44100,
            full_audio_b64="abc",
            include_full_audio=True,
        )

        self.assertEqual(msg["full_audio_b64"], "abc")

    def test_play_steps_are_clamped(self):
        self.assertEqual(server.clamp_play_steps(2), 10)
        self.assertEqual(server.clamp_play_steps(40), 40)
        self.assertEqual(server.clamp_play_steps(500), 120)
        self.assertEqual(server.clamp_play_steps("not-a-number"), server.STREAM_PLAY_STEPS)

    def test_rag_segment_buffer_flushes_sentence_and_cancel_discards_remaining_text(self):
        buffer = server.RAGSegmentBuffer()

        self.assertEqual(buffer.push("नमस्ते, "), [])
        self.assertEqual(buffer.push("मैं आपकी मदद कर सकता हूँ।"), ["नमस्ते, मैं आपकी मदद कर सकता हूँ।"])
        self.assertEqual(buffer.push("यह queued रहेगा"), [])

        buffer.cancel()

        self.assertEqual(buffer.flush(), [])
        self.assertTrue(buffer.cancelled)

    def test_split_into_sentences(self):
        self.assertEqual(server.split_into_sentences(""), [])
        self.assertEqual(server.split_into_sentences(None), [])
        self.assertEqual(server.split_into_sentences("Hello world"), ["Hello world"])
        text = "Hello world! How are you? नमस्ते। ठीक है॥ Next line\nDone."
        expected = [
            "Hello world!",
            "How are you?",
            "नमस्ते।",
            "ठीक है॥",
            "Next line",
            "Done."
        ]
        self.assertEqual(server.split_into_sentences(text), expected)


class AsyncStreamingContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_stream_audio_chunks_surfaces_generation_errors(self):
        original = server._stream_synthesize_sync

        def failing_stream(*args, **kwargs):
            raise RuntimeError("boom")
            yield

        server._stream_synthesize_sync = failing_stream
        try:
            chunks = server.stream_audio_chunks("hello", "en-IN", "default")
            with self.assertRaisesRegex(RuntimeError, "boom"):
                async for _chunk in chunks:
                    pass
        finally:
            server._stream_synthesize_sync = original

    async def test_send_streaming_synthesis_omits_full_audio_unless_requested(self):
        original_stream_audio_chunks = server.stream_audio_chunks
        original_audio_cache = server.audio_cache
        original_generation_semaphore = server._generation_semaphore

        async def fake_stream_audio_chunks(*args, **kwargs):
            yield b"chunk", 44100, False
            yield b"full", 44100, True

        server.stream_audio_chunks = fake_stream_audio_chunks
        server.audio_cache = server.LRUCache(10)
        server._generation_semaphore = asyncio.Semaphore(1)
        try:
            ws = FakeWebSocket()
            await server._send_streaming_synthesis(
                ws,
                "req_1",
                "hello",
                "en-IN",
                "default",
                include_full_audio=False,
                streaming_mode="token",
            )
            self.assertEqual([m["type"] for m in ws.messages], ["audio_chunk", "audio_end"])
            self.assertNotIn("full_audio_b64", ws.messages[-1])

            ws = FakeWebSocket()
            await server._send_streaming_synthesis(
                ws,
                "req_2",
                "hello again",
                "en-IN",
                "default",
                include_full_audio=True,
                streaming_mode="token",
            )
            self.assertIn("full_audio_b64", ws.messages[-1])
        finally:
            server.stream_audio_chunks = original_stream_audio_chunks
            server.audio_cache = original_audio_cache
            server._generation_semaphore = original_generation_semaphore

    async def test_send_streaming_synthesis_sentence_mode(self):
        original_synthesize_full = server.synthesize_full
        original_audio_cache = server.audio_cache
        original_generation_semaphore = server._generation_semaphore

        import io
        import soundfile as sf
        import numpy as np
        import base64

        dummy_arr = np.zeros(8000, dtype=np.float32)
        buf = io.BytesIO()
        sf.write(buf, dummy_arr, 8000, format="WAV", subtype="PCM_16")
        dummy_wav = buf.getvalue()

        async def fake_synthesize_full(text, lang_code, style, custom_description=None):
            return dummy_wav, 8000

        server.synthesize_full = fake_synthesize_full
        server.audio_cache = server.LRUCache(10)
        server._generation_semaphore = asyncio.Semaphore(1)
        try:
            ws = FakeWebSocket()
            await server._send_streaming_synthesis(
                ws,
                "req_sent_1",
                "Hello world. This is a test!",
                "en-IN",
                "default",
                include_full_audio=True,
                streaming_mode="sentence",
            )
            self.assertEqual(len(ws.messages), 3)
            self.assertEqual(ws.messages[0]["type"], "audio_chunk")
            self.assertEqual(ws.messages[0]["chunk_index"], 0)
            self.assertEqual(ws.messages[1]["type"], "audio_chunk")
            self.assertEqual(ws.messages[1]["chunk_index"], 1)
            self.assertEqual(ws.messages[2]["type"], "audio_end")
            self.assertEqual(ws.messages[2]["total_chunks"], 2)
            self.assertIn("full_audio_b64", ws.messages[2])

            full_audio_bytes = base64.b64decode(ws.messages[2]["full_audio_b64"])
            data, sr = sf.read(io.BytesIO(full_audio_bytes))
            self.assertEqual(sr, 8000)
            self.assertEqual(len(data), 16000)
        finally:
            server.synthesize_full = original_synthesize_full
            server.audio_cache = original_audio_cache
            server._generation_semaphore = original_generation_semaphore


class FakeWebSocket:
    def __init__(self):
        self.messages = []

    async def send(self, payload):
        self.messages.append(json.loads(payload))


if __name__ == "__main__":
    unittest.main()
